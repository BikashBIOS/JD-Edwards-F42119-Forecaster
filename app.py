import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
import os
import pickle
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="JDE F42119 Demand Forecaster",
    page_icon="📦",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 2rem; }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.5px;
    }
    .hero-sub {
        font-size: 1rem;
        color: #8B9EC7;
        margin-top: -0.5rem;
        margin-bottom: 1.5rem;
    }
    .winner-card {
        background: linear-gradient(135deg, #1a2a4a 0%, #0d1f3c 100%);
        border: 2px solid #3B82F6;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
    }
    .winner-label {
        font-size: 0.75rem;
        font-weight: 700;
        color: #3B82F6;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .winner-name {
        font-size: 2rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0.2rem 0;
    }
    .winner-mape {
        font-size: 1rem;
        color: #60A5FA;
    }
    .metric-card {
        background: #1a1f2e;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        border-left: 4px solid #3B82F6;
        margin-bottom: 0.8rem;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #8B9EC7;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #FFFFFF;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #60A5FA;
    }
    .model-card {
        background: #1a1f2e;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
        border: 1px solid #2a3347;
    }
    .model-card.best {
        border: 1.5px solid #3B82F6;
        background: #1a2a4a;
    }
    .model-name {
        font-size: 1rem;
        font-weight: 700;
        color: #FFFFFF;
    }
    .mape-pill {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .mape-best { background: #14532d; color: #4ade80; }
    .mape-mid  { background: #713f12; color: #fbbf24; }
    .mape-poor { background: #7f1d1d; color: #f87171; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #2a3347;
    }
    .jde-badge {
        background: #1e3a5f;
        color: #60A5FA;
        border-radius: 6px;
        padding: 0.2rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #2563EB, #1d4ed8);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8, #1e40af);
        transform: translateY(-1px);
    }
    div[data-testid="stProgress"] > div {
        background-color: #3B82F6 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize ma_forecasts at top level so it's always defined
ma_forecasts = {}

# ══════════════════════════════════════════════════════════════
# PIPELINE FUNCTIONS
# ══════════════════════════════════════════════════════════════

@st.cache_data
def load_and_clean(uploaded_file):
    """Load and clean raw F42119 CSV — same pipeline as Notebook 01."""
    df = pd.read_csv(uploaded_file, low_memory=False)
    df.columns = df.columns.str.strip()

    COLS_MAP = {
        'G/L Date'        : 'gl_date',
        'Quantity Shipped' : 'qty_shipped',
        'Extended Price'   : 'extended_price',
        'Ln Ty'           : 'line_type',
        'Short Item No'   : 'item_no',
        'Description'     : 'description',
        'Business Unit'   : 'business_unit',
        'Order Number'    : 'order_number'
    }
    df = df[[c for c in COLS_MAP if c in df.columns]].rename(columns=COLS_MAP)

    df['qty_shipped']    = pd.to_numeric(df['qty_shipped'].astype(str).str.replace(',',''), errors='coerce')
    df['extended_price'] = pd.to_numeric(df['extended_price'].astype(str).str.replace(',',''), errors='coerce')
    df['gl_date']        = pd.to_datetime(df['gl_date'], errors='coerce')

    df = df[df['extended_price'].notna()]
    df = df[df['line_type'].str.strip() == 'S']
    df = df[df['qty_shipped'] > 0]
    df = df[df['extended_price'] > 0]
    df = df[df['gl_date'].notna()]

    monthly = df.groupby(df['gl_date'].dt.to_period('M')).agg(
        total_qty     = ('qty_shipped', 'sum'),
        total_revenue = ('extended_price', 'sum'),
        order_count   = ('order_number', 'nunique')
    ).reset_index()
    monthly['ds'] = monthly['gl_date'].dt.to_timestamp()
    monthly = monthly.sort_values('ds').reset_index(drop=True)

    return df, monthly


def run_prophet(monthly):
    """Train Prophet and forecast next month."""
    from prophet import Prophet

    TEST_MONTHS = 3
    train = monthly.iloc[:-TEST_MONTHS]
    test  = monthly.iloc[-TEST_MONTHS:]

    results = {}
    for target, col in [('Quantity', 'total_qty'), ('Revenue', 'total_revenue')]:
        df_p = train[['ds', col]].rename(columns={col: 'y'})
        m = Prophet(
            seasonality_mode='additive',
            changepoint_prior_scale=0.3,
            yearly_seasonality=True,
            changepoints=['2023-06-01']
        )
        m.fit(df_p)

        # Test prediction
        future_test = m.make_future_dataframe(periods=TEST_MONTHS, freq='MS')
        forecast_test = m.predict(future_test)
        test_pred = forecast_test.tail(TEST_MONTHS)['yhat'].values
        test_actual = test[col].values
        mape = np.mean(np.abs((test_actual - test_pred) / test_actual)) * 100

        # Retrain on full data and forecast 1 month ahead
        df_full = monthly[['ds', col]].rename(columns={col: 'y'})
        m2 = Prophet(
            seasonality_mode='additive',
            changepoint_prior_scale=0.3,
            yearly_seasonality=True,
            changepoints=['2023-06-01']
        )
        m2.fit(df_full)
        future = m2.make_future_dataframe(periods=1, freq='MS')
        fc = m2.predict(future).iloc[-1]

        results[target] = {
            'mape'      : mape,
            'forecast'  : fc['yhat'],
            'lower'     : fc['yhat_lower'],
            'upper'     : fc['yhat_upper'],
            'test_pred' : test_pred,
            'test_actual': test_actual
        }

    return results


def run_lstm(monthly):
    """Train Stacked LSTM and forecast next month."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler

    tf.random.set_seed(42)
    np.random.seed(42)

    LOOKBACK    = 6
    TEST_MONTHS = 3
    EPOCHS      = 200
    BATCH_SIZE  = 8

    def make_sequences(data, lb):
        X, y = [], []
        for i in range(lb, len(data)):
            X.append(data[i-lb:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    def build_model(lb):
        m = Sequential([
            LSTM(64, return_sequences=True, input_shape=(lb, 1)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1)
        ])
        m.compile(optimizer='adam', loss='mse')
        return m

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=0)
    ]

    results = {}
    for target, col in [('Quantity', 'total_qty'), ('Revenue', 'total_revenue')]:
        series = monthly[col].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(series)

        X, y = make_sequences(scaled, LOOKBACK)
        X_train, X_test = X[:-TEST_MONTHS], X[-TEST_MONTHS:]
        y_train, y_test = y[:-TEST_MONTHS], y[-TEST_MONTHS:]

        X_train = X_train.reshape(-1, LOOKBACK, 1)
        X_test  = X_test.reshape(-1, LOOKBACK, 1)

        model = build_model(LOOKBACK)
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                  validation_split=0.15, callbacks=callbacks, verbose=0)

        test_pred_s  = model.predict(X_test, verbose=0)
        test_pred    = scaler.inverse_transform(test_pred_s).flatten()
        test_actual  = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
        mape = np.mean(np.abs((test_actual - test_pred) / test_actual)) * 100

        last_seq = scaled[-LOOKBACK:].reshape(1, LOOKBACK, 1)
        fc_scaled = model.predict(last_seq, verbose=0)
        forecast  = scaler.inverse_transform(fc_scaled)[0][0]

        results[target] = {
            'mape'       : mape,
            'forecast'   : forecast,
            'lower'      : None,
            'upper'      : None,
            'test_pred'  : test_pred,
            'test_actual': test_actual
        }

    return results


def run_arima(monthly):
    """Train Auto ARIMA/SARIMA and forecast next month."""
    from pmdarima import auto_arima

    TEST_MONTHS = 3
    train = monthly.iloc[:-TEST_MONTHS]
    test  = monthly.iloc[-TEST_MONTHS:]

    results = {}
    for target, col in [('Quantity', 'total_qty'), ('Revenue', 'total_revenue')]:
        model = auto_arima(
            train[col].values,
            start_p=0, max_p=5,
            start_q=0, max_q=5,
            start_P=0, max_P=2,
            start_Q=0, max_Q=2,
            d=1, D=1,
            seasonal=True, m=12,
            stepwise=False,
            suppress_warnings=True,
            error_action='ignore',
            trace=False
        )
        test_pred, conf_int = model.predict(n_periods=TEST_MONTHS, return_conf_int=True)
        test_actual = test[col].values
        mape = np.mean(np.abs((test_actual - test_pred) / test_actual)) * 100

        model.update(test_actual)
        fc, fc_conf = model.predict(n_periods=1, return_conf_int=True)

        results[target] = {
            'mape'       : mape,
            'forecast'   : fc[0],
            'lower'      : fc_conf[0][0],
            'upper'      : fc_conf[0][1],
            'test_pred'  : test_pred,
            'test_actual': test_actual
        }

    return results


def mape_color_class(mape):
    if mape < 10:
        return 'mape-best'
    elif mape < 20:
        return 'mape-mid'
    return 'mape-poor'


def format_lak(val):
    if val >= 1e9:
        return f"LAK {val/1e9:.2f}B"
    elif val >= 1e6:
        return f"LAK {val/1e6:.1f}M"
    return f"LAK {val:,.0f}"

# ══════════════════════════════════════════════════════════════
# MOVING AVERAGE BASELINE FUNCTIONS
# ══════════════════════════════════════════════════════════════

def simple_moving_average(series, window):
    """JDE-style Simple Moving Average — Balance Forward method."""
    return series.iloc[-window:].mean()

def weighted_moving_average(series, window):
    """JDE Weighted Moving Average — recent months weighted higher."""
    data    = series.iloc[-window:].values
    weights = np.arange(1, window + 1)
    return np.average(data, weights=weights)

def exponential_smoothing(series, alpha=0.3):
    """JDE Exponential Smoothing — alpha=0.3 default."""
    values   = series.values
    smoothed = [values[0]]
    for i in range(1, len(values)):
        smoothed.append(alpha * values[i] + (1 - alpha) * smoothed[-1])
    return smoothed[-1]

def compute_ma_forecasts(monthly, target_month):
    """
    Compute all MA forecasts for target month.
    Returns dict of method -> {qty, rev} forecasts.
    """
    # Filter data up to month before target
    cutoff = target_month - pd.DateOffset(months=1)
    series  = monthly[monthly['ds'] <= cutoff].copy()

    qty_series = series['total_qty']
    rev_series = series['total_revenue']

    methods = {
        'SMA-3'  : lambda s: simple_moving_average(s, 3),
        'SMA-6'  : lambda s: simple_moving_average(s, 6),
        'SMA-12' : lambda s: simple_moving_average(s, min(12, len(s))),
        'WMA-6'  : lambda s: weighted_moving_average(s, min(6, len(s))),
        'EXP'    : lambda s: exponential_smoothing(s, alpha=0.3)
    }

    results = {}
    for name, fn in methods.items():
        try:
            results[name] = {
                'qty': fn(qty_series),
                'rev': fn(rev_series)
            }
        except Exception:
            results[name] = {'qty': 0, 'rev': 0}

    return results


def forecast_chart(monthly, results_dict, target, next_month_label):
    """Plot historical + all model forecasts."""
    col = 'total_qty' if target == 'Quantity' else 'total_revenue'
    colors = {'Prophet': '#3B82F6', 'LSTM': '#F59E0B', 'ARIMA': '#10B981'}

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#1a1f2e')

    ax.plot(monthly['ds'], monthly[col],
            color='#CBD5E1', linewidth=2, marker='o', markersize=3,
            label='Historical', zorder=3)

    next_ds = monthly['ds'].max() + pd.DateOffset(months=1)
    for model_name, res in results_dict.items():
        fc = res[target]['forecast']
        ax.scatter(next_ds, fc, s=120, color=colors[model_name],
                   zorder=5, label=f'{model_name}: {fc:,.0f}' if target == 'Quantity' else f'{model_name}: {format_lak(fc)}')
        ax.plot([monthly['ds'].iloc[-1], next_ds], [monthly[col].iloc[-1], fc],
                linestyle='--', color=colors[model_name], linewidth=1.5, alpha=0.7)
        if res[target]['lower'] and res[target]['upper']:
            ax.errorbar(next_ds, fc,
                        yerr=[[fc - res[target]['lower']], [res[target]['upper'] - fc]],
                        fmt='none', color=colors[model_name], capsize=5, alpha=0.5)

    ax.axvline(next_ds - pd.DateOffset(months=1), color='#475569', linestyle=':', linewidth=1)
    ax.text(next_ds + pd.DateOffset(days=5), ax.get_ylim()[1]*0.95,
            next_month_label, color='#94A3B8', fontsize=8)

    ax.set_title(f'Monthly {target} — Historical + January 2026 Forecasts',
                 color='#FFFFFF', fontweight='bold', fontsize=12, pad=12)
    ax.tick_params(colors='#8B9EC7')
    ax.spines[:].set_color('#2a3347')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'{x:,.0f}' if target == 'Quantity' else f'{x/1e9:.1f}B'))
    legend = ax.legend(facecolor='#1a1f2e', edgecolor='#2a3347',
                       labelcolor='#CBD5E1', fontsize=9)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════

# ── Hero header ───────────────────────────────────────────────
st.markdown('<div class="hero-title">📦 JDE F42119 Demand Forecaster</div>', unsafe_allow_html=True)
st.markdown('''<div class="hero-sub">
    <span class="jde-badge">F42119</span>
    <span class="jde-badge">P4210</span>
    Prophet · LSTM · ARIMA · Real-world validated
</div>''', unsafe_allow_html=True)

st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("Upload your F42119 extract and select which models to run.")

    uploaded = st.file_uploader(
        "Upload raw_data.csv (F42119 extract)",
        type=['csv'],
        help="Raw F42119 extract — Jan 2023 to Dec 2025"
    )

    st.markdown("#### Models to Run")
    run_prophet_flag = st.checkbox("Facebook Prophet", value=True)
    run_lstm_flag    = st.checkbox("Stacked LSTM", value=True)
    run_arima_flag   = st.checkbox("Auto ARIMA / SARIMA", value=True)

    st.markdown("---")
    st.markdown("#### Optional: Validate against Actuals")
    val_file = st.file_uploader(
        "Upload jan_raw_data.csv (Jan 2026 actuals)",
        type=['csv'],
        help="Optional — validates forecast against real January 2026 data"
    )

    st.markdown("---")
    run_btn = st.button("🚀 Run Forecast Pipeline")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#475569;'>
    <b>JDE Context</b><br>
    Table: F42119<br>
    App: P4210<br>
    Filters: Ln Ty=S, Qty>0<br>
    Time axis: G/L Date (SDDGL)<br>
    Currency: LAK (Laos Kip)
    </div>
    """, unsafe_allow_html=True)

# ── Main content ──────────────────────────────────────────────
if not uploaded:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Step 1</div>
            <div class="metric-value">📂 Upload</div>
            <div class="metric-sub">Upload your F42119 CSV in the sidebar</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Step 2</div>
            <div class="metric-value">⚙️ Configure</div>
            <div class="metric-sub">Select models to run</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Step 3</div>
            <div class="metric-value">🚀 Forecast</div>
            <div class="metric-sub">Click Run Forecast Pipeline</div>
        </div>""", unsafe_allow_html=True)

    st.info("👈 Upload your raw_data.csv in the sidebar to get started.")
    st.stop()


if run_btn:
    # ── Load & clean data ──────────────────────────────────────
    with st.spinner("Loading and cleaning F42119 data..."):
        df, monthly = load_and_clean(uploaded)

    st.markdown('<div class="section-header">📊 Data Overview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Months", f"{len(monthly)}")
    c2.metric("Total Orders", f"{df['order_number'].nunique():,}")
    c3.metric("Unique Items", f"{df['item_no'].nunique()}")
    c4.metric("Date Range", f"{monthly['ds'].min().strftime('%b %Y')} – {monthly['ds'].max().strftime('%b %Y')}")

    next_month = monthly['ds'].max() + pd.DateOffset(months=1)
    next_month_label = next_month.strftime('%b %Y')

    # ── Run selected models ────────────────────────────────────
    all_results = {}
    progress = st.progress(0)
    status   = st.empty()

    if run_prophet_flag:
        status.markdown("⏳ Training **Prophet** model...")
        all_results['Prophet'] = run_prophet(monthly)
        progress.progress(33 if run_lstm_flag or run_arima_flag else 100)

    if run_lstm_flag:
        status.markdown("⏳ Training **LSTM** model (this may take 1–2 mins)...")
        all_results['LSTM'] = run_lstm(monthly)
        progress.progress(66 if run_arima_flag else 100)

    if run_arima_flag:
        status.markdown("⏳ Training **ARIMA** model (brute-force search ~5 mins)...")
        all_results['ARIMA'] = run_arima(monthly)
        progress.progress(100)

    status.empty()
    progress.empty()

    if not all_results:
        st.warning("Please select at least one model to run.")
        st.stop()
    forecast_target_month = monthly['ds'].max() + pd.DateOffset(months=1)
    ma_forecasts = compute_ma_forecasts(monthly, next_month)
    st.session_state['ma_forecasts'] = ma_forecasts

    # ── Compute average MAPE per model ─────────────────────────
    model_scores = {}
    for model_name, res in all_results.items():
        avg_mape = (res['Quantity']['mape'] + res['Revenue']['mape']) / 2
        model_scores[model_name] = avg_mape

    best_model = min(model_scores, key=model_scores.get)

    # ══════════════════════════════════════════════════════════
    # WINNER CARD
    # ══════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">🏆 Best Model</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="winner-card">
        <div class="winner-label">🏆 Best Performing Model (Lowest Test MAPE)</div>
        <div class="winner-name">{best_model}</div>
        <div class="winner-mape">
            Avg MAPE: {model_scores[best_model]:.2f}% &nbsp;|&nbsp;
            Qty MAPE: {all_results[best_model]['Quantity']['mape']:.2f}% &nbsp;|&nbsp;
            Rev MAPE: {all_results[best_model]['Revenue']['mape']:.2f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # JANUARY 2026 FORECAST — BEST MODEL
    # ══════════════════════════════════════════════════════════
    st.markdown(f'<div class="section-header">🔮 {next_month_label} Forecast — {best_model}</div>',
                unsafe_allow_html=True)

    best_res = all_results[best_model]
    fc1, fc2 = st.columns(2)

    with fc1:
        qty_fc = best_res['Quantity']['forecast']
        qty_lo = best_res['Quantity']['lower']
        qty_hi = best_res['Quantity']['upper']
        range_str = f"{qty_lo:,.0f} – {qty_hi:,.0f}" if qty_lo else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Forecasted Quantity</div>
            <div class="metric-value">{qty_fc:,.0f} <span style='font-size:1rem;color:#8B9EC7'>units</span></div>
            <div class="metric-sub">95% range: {range_str}</div>
        </div>""", unsafe_allow_html=True)

    with fc2:
        rev_fc = best_res['Revenue']['forecast']
        rev_lo = best_res['Revenue']['lower']
        rev_hi = best_res['Revenue']['upper']
        range_str = f"{format_lak(rev_lo)} – {format_lak(rev_hi)}" if rev_lo else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Forecasted Revenue</div>
            <div class="metric-value">{format_lak(rev_fc)}</div>
            <div class="metric-sub">95% range: {range_str}</div>
        </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # ALL MODELS COMPARISON
    # ══════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">📋 All Models Comparison</div>', unsafe_allow_html=True)

    cols = st.columns(len(all_results))
    for i, (model_name, res) in enumerate(all_results.items()):
        avg_mape = model_scores[model_name]
        is_best  = model_name == best_model
        cc       = mape_color_class(avg_mape)
        badge    = "🏆 Best" if is_best else ""

        with cols[i]:
            st.markdown(f"""
            <div class="model-card {'best' if is_best else ''}">
                <div class="model-name">{model_name} {badge}</div>
                <br>
                <div class="metric-label">Jan {next_month.year} Quantity</div>
                <div style='color:#FFFFFF;font-weight:600;'>{res['Quantity']['forecast']:,.0f} units</div>
                <br>
                <div class="metric-label">Jan {next_month.year} Revenue</div>
                <div style='color:#FFFFFF;font-weight:600;'>{format_lak(res['Revenue']['forecast'])}</div>
                <br>
                <div class="metric-label">Test Qty MAPE</div>
                <span class="mape-pill {mape_color_class(res['Quantity']['mape'])}">{res['Quantity']['mape']:.2f}%</span>
                <br><br>
                <div class="metric-label">Test Rev MAPE</div>
                <span class="mape-pill {mape_color_class(res['Revenue']['mape'])}">{res['Revenue']['mape']:.2f}%</span>
                <br><br>
                <div class="metric-label">Avg MAPE</div>
                <span class="mape-pill {cc}">{avg_mape:.2f}%</span>
            </div>
            """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # FORECAST CHARTS
    # ══════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">📈 Forecast Charts</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📦 Quantity", "💰 Revenue"])
    with tab1:
        fig = forecast_chart(monthly, all_results, 'Quantity', next_month_label)
        st.pyplot(fig)
    with tab2:
        fig = forecast_chart(monthly, all_results, 'Revenue', next_month_label)
        st.pyplot(fig)

    # ══════════════════════════════════════════════════════════
    # REAL VALIDATION (if Jan 2026 actuals uploaded)
    # ══════════════════════════════════════════════════════════
    if val_file:
        st.markdown('<div class="section-header">✅ Real World Validation — January 2026 Actuals</div>',
                    unsafe_allow_html=True)

        df_val, _ = load_and_clean(val_file)
        actual_qty = df_val['qty_shipped'].sum()
        actual_rev = df_val['extended_price'].sum()

        v1, v2 = st.columns(2)
        with v1:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color:#10B981;">
                <div class="metric-label">Actual Jan 2026 Quantity</div>
                <div class="metric-value">{actual_qty:,.0f} <span style='font-size:1rem;color:#8B9EC7'>units</span></div>
            </div>""", unsafe_allow_html=True)
        with v2:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color:#10B981;">
                <div class="metric-label">Actual Jan 2026 Revenue</div>
                <div class="metric-value">{format_lak(actual_rev)}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("#### Forecast vs Actual")
        val_rows = []
        for model_name, res in all_results.items():
            qty_fc  = res['Quantity']['forecast']
            rev_fc  = res['Revenue']['forecast']
            qty_err = (qty_fc - actual_qty) / actual_qty * 100
            rev_err = (rev_fc - actual_rev) / actual_rev * 100
            qty_mape = abs(qty_err)
            rev_mape = abs(rev_err)
            val_rows.append({
                'Model'         : model_name,
                'Pred Qty'      : f"{qty_fc:,.0f}",
                'Qty Error'     : f"{qty_err:+.1f}%",
                'Qty MAPE'      : qty_mape,
                'Pred Rev'      : format_lak(rev_fc),
                'Rev Error'     : f"{rev_err:+.1f}%",
                'Rev MAPE'      : rev_mape,
                'Avg MAPE'      : (qty_mape + rev_mape) / 2
            })

        # Ensemble
        ens_qty = np.mean([all_results[m]['Quantity']['forecast'] for m in all_results])
        ens_rev = np.mean([all_results[m]['Revenue']['forecast'] for m in all_results])
        ens_qe  = (ens_qty - actual_qty) / actual_qty * 100
        ens_re  = (ens_rev - actual_rev) / actual_rev * 100
        val_rows.append({
            'Model'   : 'Ensemble',
            'Pred Qty': f"{ens_qty:,.0f}",
            'Qty Error': f"{ens_qe:+.1f}%",
            'Qty MAPE': abs(ens_qe),
            'Pred Rev': format_lak(ens_rev),
            'Rev Error': f"{ens_re:+.1f}%",
            'Rev MAPE': abs(ens_re),
            'Avg MAPE': (abs(ens_qe) + abs(ens_re)) / 2
        })

        val_df = pd.DataFrame(val_rows).sort_values('Avg MAPE')
        real_winner = val_df.iloc[0]['Model']

        st.dataframe(
            val_df[['Model','Pred Qty','Qty Error','Pred Rev','Rev Error','Avg MAPE']].reset_index(drop=True),
            use_container_width=True
        )

        st.success(f"🏆 **Real-world winner: {real_winner}** with {val_df.iloc[0]['Avg MAPE']:.2f}% average MAPE on actual January 2026 data")

        # ══════════════════════════════════════════════════════════
        # MOVING AVERAGE BASELINE COMPARISON
        # ══════════════════════════════════════════════════════════
        st.markdown('<div class="section-header">📉 JDE Baseline — Moving Average Comparison</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div style='background:#1a1f2e;border-radius:10px;padding:1rem 1.2rem;
        border-left:4px solid #F59E0B;margin-bottom:1rem;color:#CBD5E1;font-size:0.85rem;'>
        <b>What is this?</b> JDE's native forecasting uses Simple Moving Average (SMA) —
        no ML, no seasonality handling, no changepoint detection.
        We benchmark our ML models against all JDE-equivalent MA methods
        on the <b>exact same forecast target</b> to quantify the real improvement.
        </div>
        """, unsafe_allow_html=True)

        # Build comparison table — read from session state
        ma_data = st.session_state.get('ma_forecasts', {})

        ma_rows = []
        for ma_name, ma_vals in ma_data.items():
            ma_rows.append({
                'Method'  : f'{ma_name} (JDE Baseline)',
                'Pred Qty': f"{ma_vals['qty']:,.0f}",
                'Pred Rev': format_lak(ma_vals['rev']),
                'Qty_raw' : ma_vals['qty'],
                'Rev_raw' : ma_vals['rev'],
                'Type'    : 'MA Baseline'
            })

        # Add ML models
        for model_name, res in all_results.items():
            ma_rows.append({
                'Method'  : f'{model_name} {"🏆" if model_name == best_model else ""}',
                'Pred Qty': f"{res['Quantity']['forecast']:,.0f}",
                'Pred Rev': format_lak(res['Revenue']['forecast']),
                'Qty_raw' : res['Quantity']['forecast'],
                'Rev_raw' : res['Revenue']['forecast'],
                'Type'    : 'ML Model'
            })

        if ma_rows and ma_data:
            # MA vs ML chart
            fig_ma, axes_ma = plt.subplots(1, 2, figsize=(16, 6))
            fig_ma.patch.set_facecolor('#0f1117')
            fig_ma.suptitle(
                f'ML Models vs JDE Moving Average Baselines — {next_month_label}',
                fontsize=13, fontweight='bold', color='#FFFFFF'
            )

            ml_color  = '#3B82F6'
            ma_color_list = ['#EF4444','#EC4899','#8B5CF6','#F97316','#6B7280']

            all_names  = [r['Method'] for r in ma_rows]
            all_qty    = [r['Qty_raw'] for r in ma_rows]
            all_rev    = [r['Rev_raw'] for r in ma_rows]
            bar_colors = [
                ml_color if r['Type'] == 'ML Model'
                else ma_color_list[i % len(ma_color_list)]
                for i, r in enumerate(ma_rows)
            ]

            x = np.arange(len(all_names))

            for ax in axes_ma:
                ax.set_facecolor('#1a1f2e')
                ax.tick_params(colors='#8B9EC7')
                ax.spines[:].set_color('#2a3347')

            axes_ma[0].bar(x, all_qty, color=bar_colors, alpha=0.85, width=0.6)
            axes_ma[0].set_title('Quantity Forecast', color='#FFFFFF', fontweight='bold')
            axes_ma[0].set_ylabel('Quantity (Units)', color='#8B9EC7')
            axes_ma[0].set_xticks(x)
            axes_ma[0].set_xticklabels(all_names, rotation=30, ha='right',
                                        fontsize=8, color='#CBD5E1')
            axes_ma[0].yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))

            axes_ma[1].bar(x, [r/1e9 for r in all_rev], color=bar_colors, alpha=0.85, width=0.6)
            axes_ma[1].set_title('Revenue Forecast (LAK)', color='#FFFFFF', fontweight='bold')
            axes_ma[1].set_ylabel('Revenue (LAK Billions)', color='#8B9EC7')
            axes_ma[1].set_xticks(x)
            axes_ma[1].set_xticklabels(all_names, rotation=30, ha='right',
                                        fontsize=8, color='#CBD5E1')
            axes_ma[1].yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f'{v:.0f}B'))

            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=ml_color,   label='ML Models'),
                Patch(facecolor='#EF4444', label='JDE MA Baselines')
            ]
            axes_ma[1].legend(handles=legend_elements, facecolor='#1a1f2e',
                            edgecolor='#2a3347', labelcolor='#CBD5E1', fontsize=9)

            plt.tight_layout()
            st.pyplot(fig_ma)

            # Forecast comparison table
            st.markdown("#### 📊 Forecast Comparison Table")
            display_df = pd.DataFrame([{
                'Method'  : r['Method'],
                'Type'    : r['Type'],
                'Pred Qty': r['Pred Qty'],
                'Pred Rev': r['Pred Rev']
            } for r in ma_rows])
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        else:
            st.info("MA baselines will appear after running the forecast pipeline.")

        # Error vs actuals — only when validation file uploaded
        if val_file:
            actual_qty_val = df_val['qty_shipped'].sum()
            actual_rev_val = df_val['extended_price'].sum()

            st.markdown("#### 🎯 Error vs January 2026 Actuals")

            comparison_rows = []

            # ML models
            for model_name, res in all_results.items():
                qty_fc   = res['Quantity']['forecast']
                rev_fc   = res['Revenue']['forecast']
                qty_mape = abs((qty_fc - actual_qty_val) / actual_qty_val * 100)
                rev_mape = abs((rev_fc - actual_rev_val) / actual_rev_val * 100)
                comparison_rows.append({
                    'Model'    : f'{model_name} {"🏆" if model_name == best_model else ""}',
                    'Type'     : 'ML Model',
                    'Qty MAPE' : round(qty_mape, 2),
                    'Rev MAPE' : round(rev_mape, 2),
                    'Avg MAPE' : round((qty_mape + rev_mape) / 2, 2)
                })

            # MA baselines
            for ma_name, ma_vals in st.session_state.get('ma_forecasts', {}).items():
                qty_mape = abs((ma_vals['qty'] - actual_qty_val) / actual_qty_val * 100)
                rev_mape = abs((ma_vals['rev'] - actual_rev_val) / actual_rev_val * 100)
                comparison_rows.append({
                    'Model'    : f'{ma_name} (JDE)',
                    'Type'     : 'MA Baseline',
                    'Qty MAPE' : round(qty_mape, 2),
                    'Rev MAPE' : round(rev_mape, 2),
                    'Avg MAPE' : round((qty_mape + rev_mape) / 2, 2)
                })

            if comparison_rows:
                comp_df  = pd.DataFrame(comparison_rows).sort_values('Avg MAPE')
                ml_rows  = comp_df[comp_df['Type'] == 'ML Model']
                ma_rows_f = comp_df[comp_df['Type'] == 'MA Baseline']

                st.dataframe(comp_df.reset_index(drop=True),
                            use_container_width=True, hide_index=True)

                if not ml_rows.empty and not ma_rows_f.empty:
                    best_ml_mape = ml_rows['Avg MAPE'].min()
                    best_ma_mape = ma_rows_f['Avg MAPE'].min()
                    best_ma_name = ma_rows_f.sort_values('Avg MAPE').iloc[0]['Model']
                    improvement  = ((best_ma_mape - best_ml_mape) / best_ma_mape) * 100

                    st.markdown(f"""
                    <div class="winner-card">
                        <div class="winner-label">📉 ML vs JDE Baseline Improvement</div>
                        <div class="winner-name">{improvement:.1f}% Error Reduction</div>
                        <div class="winner-mape">
                            Best ML: {best_ml_mape:.2f}% avg MAPE &nbsp;|&nbsp;
                            Best JDE Baseline ({best_ma_name}): {best_ma_mape:.2f}% avg MAPE
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Run the forecast pipeline first to see ML vs JDE baseline comparison.")

    # ══════════════════════════════════════════════════════════
    # DOWNLOAD RESULTS
    # ══════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">💾 Download Results</div>', unsafe_allow_html=True)

    summary_rows = []
    for model_name, res in all_results.items():
        summary_rows.append({
            'Model'           : model_name,
            'Jan2026_Qty'     : res['Quantity']['forecast'],
            'Jan2026_Rev_LAK' : res['Revenue']['forecast'],
            'Test_Qty_MAPE'   : res['Quantity']['mape'],
            'Test_Rev_MAPE'   : res['Revenue']['mape'],
            'Avg_MAPE'        : model_scores[model_name],
            'Is_Best'         : model_name == best_model
        })

    # Add MA baselines to download
    for ma_name, ma_vals in ma_forecasts.items():
        summary_rows.append({
            'Model'           : f'{ma_name} (JDE Baseline)',
            'Type'            : 'MA Baseline',
            'Jan2026_Qty'     : ma_vals['qty'],
            'Jan2026_Rev_LAK' : ma_vals['rev'],
            'Test_Qty_MAPE'   : None,
            'Test_Rev_MAPE'   : None,
            'Avg_MAPE'        : None,
            'Is_Best'         : False
        })

    summary_df = pd.DataFrame(summary_rows)
    csv = summary_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="⬇️ Download Forecast Summary CSV",
        data=csv,
        file_name=f"jde_f42119_forecast_{next_month_label.replace(' ','_')}.csv",
        mime='text/csv'
    )

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; color:#475569; font-size:0.8rem; padding: 1rem 0;'>
        JDE F42119 Demand Forecasting · Prophet vs LSTM vs ARIMA<br>
        Built with ❤️ by Bikashics · IBM India → Data Science
    </div>
    """, unsafe_allow_html=True)