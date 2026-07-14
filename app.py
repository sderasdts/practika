import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

st.set_page_config(page_title="Рекомендации по питанию", layout="wide")

st.title("Система рекомендаций по питанию")
st.write("Выберите цель и блюдо, чтобы узнать, подходит ли оно вам, и получить альтернативы.")

DATA_PATH = 'pr_dataset.csv'
MODELS_PATH = 'russian_food_models.pkl'
SCALER_PATH = 'russian_food_scaler.pkl'

FEATURES = ['calories', 'protein', 'carbs', 'fat', 'fiber', 'sugar',
            'sodium', 'cholesterol', 'cat_encoded', 'meal_encoded']


@st.cache_data
def load_dataset():
    if not os.path.exists(DATA_PATH):
        st.error(f"Файл {DATA_PATH} не найден! Сначала выполните Этап 2 в ноутбуке.")
        return None
    df = pd.read_csv(DATA_PATH).dropna().reset_index(drop=True)
    return df


@st.cache_resource
def get_models(df):
    need_retrain = False
    
    if os.path.exists(MODELS_PATH) and os.path.exists(SCALER_PATH):
        try:
            scaler_check = joblib.load(SCALER_PATH)
            if hasattr(scaler_check, 'feature_names_in_'):
                saved_features = list(scaler_check.feature_names_in_)
                if set(saved_features) != set(FEATURES):
                    need_retrain = True
        except Exception:
            need_retrain = True
            
    if need_retrain or not (os.path.exists(MODELS_PATH) and os.path.exists(SCALER_PATH)):
        if need_retrain:
            try:
                os.remove(MODELS_PATH)
                os.remove(SCALER_PATH)
            except OSError:
                pass
                
        X = df[FEATURES]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        models = {}
        targets = ['label_loss', 'label_balance', 'label_gain']
        for target in targets:
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_scaled, df[target])
            models[target] = rf

        joblib.dump(models, MODELS_PATH)
        joblib.dump(scaler, SCALER_PATH)
    else:
        models = joblib.load(MODELS_PATH)
        scaler = joblib.load(SCALER_PATH)

    return models, scaler


df = load_dataset()

if df is not None:
    models, scaler = get_models(df)

    col1, col2 = st.columns(2)

    with col1:
        goal_options = {
            'Похудение': 'label_loss',
            'Баланс': 'label_balance',
            'Набор массы': 'label_gain'
        }
        selected_goal_name = st.selectbox("Ваша цель:", list(goal_options.keys()))
        target_col = goal_options[selected_goal_name]

    with col2:
        dish_list = sorted(df['name'].unique())
        selected_dish = st.selectbox("Выберите блюдо:", dish_list)

    if st.button("Получить рекомендацию"):
        dish_row = df[df['name'] == selected_dish].iloc[0]

        input_data = pd.DataFrame([dish_row[FEATURES]])
        input_scaled = scaler.transform(input_data)
        prediction = models[target_col].predict(input_scaled)[0]

        st.divider()
        st.subheader(f"Результат для: {selected_dish}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Калории", f"{int(dish_row['calories'])} ккал")
        c2.metric("Белки", f"{dish_row['protein']} г")
        c3.metric("Жиры", f"{dish_row['fat']} г")
        c4.metric("Углеводы", f"{dish_row['carbs']} г")

        if prediction == 1:
            st.success("Блюдо отлично подходит для цели!")
        else:
            st.warning("Блюдо не соответствует цели.")

            category = dish_row['category']
            candidates = df[(df['category'] == category) & (df['name'] != selected_dish)].copy()

            if not candidates.empty:
                if target_col == 'label_loss':
                    candidates = candidates.sort_values(by='calories', ascending=True)
                elif target_col == 'label_gain':
                    candidates = candidates.sort_values(by='protein', ascending=False)
                elif target_col == 'label_balance':
                    median_cal = df['calories'].median()
                    candidates['diff'] = (candidates['calories'] - median_cal).abs()
                    candidates = candidates.sort_values(by='diff', ascending=True)

                top_pool = candidates.head(10)
                n_samples = min(10, len(top_pool))
                top_replacements = top_pool.sample(n=n_samples)

                st.info("Рекомендуем обратить внимание на эти блюда:")

                for i, (_, match) in enumerate(top_replacements.iterrows()):
                    with st.expander(f"Вариант {i + 1}: {match['name']}"):
                        r1, r2, r3, r4 = st.columns(4)

                        r1.metric("Калории", f"{int(match['calories'])}")
                        r2.metric("Белки", f"{match['protein']}г")
                        r3.metric("Жиры", f"{match['fat']}г")
                        r4.metric("Углеводы", f"{match['carbs']}г")

            else:
                st.write("К сожалению, подходящей замены в этой категории не найдено.")