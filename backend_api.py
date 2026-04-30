import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
import psycopg2
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Configuration
GEMINI_API_KEY = os.environ.get("AIzaSyDEsfBCeOjylx_rocukJprD6ZYo8Z3XzFg")
CONNECTION_STRING = "postgresql://neondb_owner:npg_mdzpYTK3LAH1@ep-super-bonus-ammuoe7v-pooler.c-5.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
client = genai.Client(api_key=GEMINI_API_KEY)


@app.route('/oracle', methods=['POST'])
def oracle_query():
    data = request.json
    user_query = data.get('query')

    system_instruction = """
    You are a ZPSL SQL generator. Respond ONLY with valid PostgreSQL.
    Schema:
    - Stadiums (stadium_id, stadium_name, city, capacity)
    - Teams (team_id, team_name, founded_year, home_stadium_id)
    - Matches (match_id, match_date, home_team_id, away_team_id, home_goals, away_goals, stadium_id, season)
    Return ONLY the SQL query. No markdown.
    """

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=user_query,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0
            )
        )
        sql_query = response.text.strip().replace("```sql", "").replace("```", "")

        conn = psycopg2.connect(CONNECTION_STRING)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        conn.close()

        formatted_results = [dict(zip(column_names, row)) for row in results]
        return jsonify({"sql": sql_query, "data": formatted_results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict_match():
    data = request.json
    home_team = data.get('home_team')
    away_team = data.get('away_team')

    try:
        # ML Logic
        conn = psycopg2.connect(CONNECTION_STRING)
        cursor = conn.cursor()
        cursor.execute("SELECT home_team_id, away_team_id, home_goals, away_goals FROM Matches;")
        records = cursor.fetchall()
        conn.close()

        df = pd.DataFrame(records, columns=['home_team_id', 'away_team_id', 'home_goals', 'away_goals'])
        df['match_result'] = df.apply(lambda row: 1 if row['home_goals'] > row['away_goals'] else (
            2 if row['home_goals'] < row['away_goals'] else 0), axis=1)

        X = df[['home_team_id', 'away_team_id']]
        y = df['match_result']
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        team_map = {"Dynamos FC": 1, "Highlanders FC": 2, "CAPS United": 3}
        upcoming_match = pd.DataFrame({'home_team_id': [team_map[home_team]], 'away_team_id': [team_map[away_team]]})
        prediction = model.predict(upcoming_match)[0]

        if prediction == 1:
            outcome_text = f"{home_team} is predicted to win."
        elif prediction == 2:
            outcome_text = f"{away_team} is predicted to win."
        else:
            outcome_text = "The match is predicted to result in a draw."

        # GenAI Narrative
        prompt = f"""
        You are a premier sports data analyst covering the Zimbabwe Premier Soccer League.
        Write a professional, 150-word pre-match analysis for {home_team} vs {away_team}.
        Our internal machine learning model predicted: {outcome_text}.
        Maintain a highly analytical, corporate sports journalism tone.
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=prompt
        )

        return jsonify({"prediction": outcome_text, "narrative": response.text.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000, debug=True)