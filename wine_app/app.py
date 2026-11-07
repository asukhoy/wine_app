import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

APP_DIR = Path(__file__).parent

FEATURE_LABELS = {
    'fixed_acidity': 'Фиксированная кислотность (г/дм³)',
    'volatile_acidity': 'Летучая кислотность (г/дм³)',
    'citric_acid': 'Лимонная кислота (г/дм³)',
    'residual_sugar': 'Остаточный сахар (г/дм³)',
    'chlorides': 'Хлориды (г/дм³)',
    'free_sulfur_dioxide': 'Свободный SO₂ (мг/дм³)',
    'total_sulfur_dioxide': 'Общий SO₂ (мг/дм³)',
    'density': 'Плотность (г/см³)',
    'pH': 'pH',
    'sulphates': 'Сульфаты (г/дм³)',
    'alcohol': 'Алкоголь (% об.)',
    'is_red': 'Тип вина',
}
FEATURE_FORMAT = {'chlorides': '%.4f', 'density': '%.5f', 'volatile_acidity': '%.3f',
                  'citric_acid': '%.3f', 'sulphates': '%.3f', 'pH': '%.2f'}


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_config():
    return json.loads((APP_DIR / 'config.json').read_text())


@st.cache_data
def load_test():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


model = load_model()
config = load_config()
test_df = load_test()

FEATURES = list(getattr(model, 'feature_names_in_', test_df.drop(columns='target').columns))
X_test = test_df[FEATURES]
y_test = test_df['target'].to_numpy()
proba_test = test_proba(model, X_test)

st.set_page_config(page_title='Подборка вин', layout='wide')
st.title('Отбор хороших вин для подписки')
st.caption('Логистическая регрессия (StandardScaler + LogisticRegression) по результатам '
           'лабораторного анализа. «Хорошее» вино — экспертная оценка не ниже 7 из 10.')

default_t = round(float(config.get('threshold', 0.5)), 3)
if 'threshold' not in st.session_state:
    st.session_state['threshold'] = default_t
with st.sidebar:
    st.header('Порог решения')
    threshold = st.slider('Вино попадает в подборку, если P(хорошее) ≥ порога',
                          min_value=0.0, max_value=1.0,
                          step=0.001, format='%.3f', key='threshold')
    st.caption(f'Порог по умолчанию из config.json: **{default_t:.3f}** '
               '(подобран на валидации по максимуму F1).')
    st.button('Вернуть порог по умолчанию',
              on_click=lambda: st.session_state.update(threshold=default_t))

tab_one, tab_metrics, tab_batch, tab_coef = st.tabs(
    ['Оценка одного вина', 'Модельные метрики', 'Пакетное предсказание', 'Что выучила модель'])

with tab_one:
    st.subheader('Введите результаты анализа')
    stats = X_test.agg(['min', 'max', 'median'])
    values = {}
    cols = st.columns(3)
    for i, f in enumerate(FEATURES):
        with cols[i % 3]:
            if f == 'is_red':
                kind = st.radio(FEATURE_LABELS[f], ['Белое', 'Красное'], horizontal=True,
                                index=int(stats.loc['median', f] >= 0.5))
                values[f] = 1 if kind == 'Красное' else 0
            else:
                lo, hi, med = (float(stats.loc[k, f]) for k in ('min', 'max', 'median'))
                values[f] = st.number_input(
                    FEATURE_LABELS.get(f, f), min_value=lo, max_value=hi, value=med,
                    step=(hi - lo) / 100, format=FEATURE_FORMAT.get(f, '%.2f'), key=f'in_{f}')

    x_one = pd.DataFrame([values], columns=FEATURES)
    p_one = float(model.predict_proba(x_one)[0, 1])
    good = p_one >= threshold

    c1, c2 = st.columns(2)
    c1.metric('Вероятность, что вино хорошее', f'{p_one:.1%}')
    c2.metric('Решение модели', 'В подборку' if good else 'Не рекомендовать',
              delta=f'{p_one - threshold:+.3f} к порогу')
    st.progress(min(max(p_one, 0.0), 1.0))
    st.caption(f'Диапазоны полей — от минимума до максимума по тестовой выборке, '
               f'значения по умолчанию — медианы. Текущий порог: {threshold:.3f}.')

with tab_metrics:
    st.subheader(f'Качество на тестовой выборке ({len(y_test)} вин) при пороге {threshold:.3f}')
    pred = (proba_test >= threshold).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr_t = fp / (fp + tn) if fp + tn else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric('Precision', f'{prec:.3f}', help='Доля хороших среди отобранных в подборку')
    m2.metric('Recall', f'{rec:.3f}', help='Доля найденных среди всех хороших вин')
    m3.metric('F1', f'{f1:.3f}')
    m4.metric('Вин в подборке', f'{tp + fp} ({(tp + fp) / len(y_test):.0%})')

    g1, g2, g3 = st.columns(3)
    with g1:
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
        metrics.ConfusionMatrixDisplay(np.array([[tn, fp], [fn, tp]]),
                                       display_labels=['обычное', 'хорошее']).plot(
            ax=ax, cmap='Blues', colorbar=False, values_format='d')
        ax.set_xlabel('Предсказание')
        ax.set_ylabel('Истина')
        ax.set_title('Матрица ошибок')
        st.pyplot(fig)
        plt.close(fig)
    with g2:
        fpr, tpr, _ = metrics.roc_curve(y_test, proba_test)
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
        ax.plot(fpr, tpr, label=f'ROC-AUC = {metrics.roc_auc_score(y_test, proba_test):.3f}')
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.scatter([fpr_t], [rec], s=90, color='crimson', edgecolor='k', zorder=5,
                   label=f'порог {threshold:.3f}')
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR (recall)')
        ax.set_title('ROC-кривая')
        ax.legend(loc='lower right')
        st.pyplot(fig)
        plt.close(fig)
    with g3:
        pr_p, pr_r, _ = metrics.precision_recall_curve(y_test, proba_test)
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
        ax.plot(pr_r, pr_p,
                label=f'PR-AUC = {metrics.average_precision_score(y_test, proba_test):.3f}')
        ax.axhline(y_test.mean(), color='k', ls='--', lw=1, label='случайная модель')
        if tp + fp:
            ax.scatter([rec], [prec], s=90, color='crimson', edgecolor='k', zorder=5,
                       label=f'порог {threshold:.3f}')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('PR-кривая')
        ax.legend(loc='upper right')
        st.pyplot(fig)
        plt.close(fig)

    st.info(
        '**Для сомелье.** Порог — это минимальная уверенность модели, при которой вино '
        'попадает в подборку.\n\n'
        '- **Порог выше** → в подборку попадают только вина, в которых модель уверена: '
        'посредственных вин меньше (precision растёт), но подборка короче, и часть '
        'хороших вин теряется (recall падает).\n'
        '- **Порог ниже** → подборка шире, хороших вин находится больше (recall растёт), '
        'но вместе с ними чаще попадают и посредственные (precision падает).\n\n'
        f'Сейчас из {int(tp + fp)} отобранных вин хорошие — {int(tp)}, '
        f'а {int(fn)} хороших вин в подборку не попали.')

with tab_batch:
    st.subheader('Оценка списка вин из CSV')
    st.write(f'Загрузите CSV со столбцами: `{", ".join(FEATURES)}`. '
             f'Используется порог из боковой панели: **{threshold:.3f}**.')
    st.download_button('Скачать пример файла (10 вин из теста)',
                       X_test.head(10).to_csv(index=False).encode('utf-8'),
                       file_name='wine_example.csv', mime='text/csv')
    uploaded = st.file_uploader('CSV-файл с признаками вин', type=['csv'])
    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f'Не удалось прочитать CSV: {e}')
            batch = None

        if batch is not None:
            missing = [c for c in FEATURES if c not in batch.columns]
            if batch.empty:
                st.error('Файл не содержит ни одной строки.')
            elif missing:
                st.error('В файле не хватает столбцов: ' + ', '.join(f'`{c}`' for c in missing)
                         + '. Добавьте их и загрузите файл снова.')
            else:
                Xb = batch[FEATURES].apply(pd.to_numeric, errors='coerce')
                bad = Xb.isna().any(axis=1)
                if bad.all():
                    st.error('Ни в одной строке нет корректных числовых значений всех признаков.')
                else:
                    if bad.any():
                        st.warning(f'{int(bad.sum())} строк(и) содержат пропуски или нечисловые '
                                   'значения — для них предсказание не построено.')
                    out = batch.copy()
                    out['proba_good'] = np.nan
                    out.loc[~bad, 'proba_good'] = model.predict_proba(Xb[~bad])[:, 1]
                    out['decision'] = np.where(bad, 'ошибка в данных',
                                               np.where(out['proba_good'] >= threshold,
                                                        'в подборку', 'не рекомендовать'))
                    if 'target' in out.columns:
                        st.caption('Столбец `target` в файле есть — он не используется для предсказания.')
                    n_sel = int((out['decision'] == 'в подборку').sum())
                    st.success(f'Оценено вин: {int((~bad).sum())}, в подборку: {n_sel}.')
                    st.dataframe(out.style.format({'proba_good': '{:.3f}'}), width='stretch')
                    st.download_button('Скачать результат (CSV)',
                                       out.to_csv(index=False).encode('utf-8'),
                                       file_name='wine_predictions.csv', mime='text/csv')

with tab_coef:
    st.subheader('Коэффициенты логистической регрессии')
    lr = model[-1]
    coef = pd.Series(lr.coef_.ravel(), index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([FEATURE_LABELS.get(f, f) for f in coef.index], coef.values,
            color=np.where(coef.values > 0, '#2a9d8f', '#e76f51'))
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel('Коэффициент (признаки стандартизованы)')
    ax.set_title('Зелёные — «за» хорошее вино, оранжевые — «против»')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    pos = coef[coef > 0].sort_values(ascending=False).head(3)
    neg = coef[coef < 0].head(3)
    st.markdown(
        '**Сильнее всего «голосуют» за хорошее вино:** '
        + ', '.join(f'{FEATURE_LABELS.get(f, f).split(" (")[0].lower()} ({v:+.2f})' for f, v in pos.items())
        + '.\n\n**Сильнее всего «против»:** '
        + ', '.join(f'{FEATURE_LABELS.get(f, f).split(" (")[0].lower()} ({v:+.2f})' for f, v in neg.items())
        + '.')
    st.caption('Признаки стандартизованы, поэтому коэффициент показывает, насколько меняются '
               'логарифмические шансы «хорошего» вина при росте признака на одно стандартное '
               'отклонение (при прочих равных). Признаки коррелируют между собой (например, '
               'плотность зависит от алкоголя и сахара), поэтому отдельные коэффициенты нельзя '
               'трактовать как независимое влияние каждого признака.')
