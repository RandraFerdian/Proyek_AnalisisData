import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="E-Commerce Analysis Dashboard", page_icon="🛒", layout="wide")

# Tetapan Gaya Visualisasi
sns.set(style='darkgrid')

# --- FUNGSI PEMBANTU (DATA PREPARATION) ---
def create_daily_orders_df(df):
    daily_orders_df = df.resample(rule='D', on='order_purchase_timestamp').agg({
        "order_id": "nunique",
        "price": "sum"
    }).reset_index()
    daily_orders_df.rename(columns={"order_id": "order_count", "price": "revenue"}, inplace=True)
    return daily_orders_df

def create_product_perf_df(df):
    product_perf = df.groupby("product_category_name").agg({
        "order_id": "nunique",
        "price": "sum",
        "review_score": "mean"
    }).sort_values(by="price", ascending=False).reset_index()
    return product_perf

def create_geography_df(df):
    state_df = df.groupby("customer_state_x").order_id.nunique().sort_values(ascending=False).reset_index()
    city_df = df.groupby("customer_city_x").order_id.nunique().sort_values(ascending=False).head(15).reset_index()
    return state_df, city_df

def create_rfm_df(df):
    rfm_df = df.groupby(by="customer_unique_id_x", as_index=False).agg({
        "order_purchase_timestamp": "max",
        "order_id": "nunique",
        "price": "sum"
    })
    rfm_df.columns = ["customer_id", "max_order_timestamp", "frequency", "monetary"]
    rfm_df["max_order_timestamp"] = pd.to_datetime(rfm_df["max_order_timestamp"]).dt.date
    recent_date = df["order_purchase_timestamp"].dt.date.max()
    rfm_df["recency"] = rfm_df["max_order_timestamp"].apply(lambda x: (recent_date - x).days)
    return rfm_df

# --- LOADING DATA ---
# Pastikan main_data.csv berada dalam folder yang sama atau nyatakan path yang betul
@st.cache_data
def load_data():
    df = pd.read_csv("main_data.csv")
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    return df

all_df = load_data()

# --- SIDEBAR FILTERS ---
with st.sidebar:
    st.title("Rentang Waktu Data")
    
    start_date, end_date = st.date_input(
        label='Rentang Masa',
        min_value=all_df["order_purchase_timestamp"].min().date(),
        max_value=all_df["order_purchase_timestamp"].max().date(),
        value=[all_df["order_purchase_timestamp"].min().date(), all_df["order_purchase_timestamp"].max().date()]
    )

# Filter Data Berdasarkan Tarikh
main_df = all_df[(all_df["order_purchase_timestamp"] >= str(start_date)) & 
                (all_df["order_purchase_timestamp"] <= str(end_date))]

# Sediakan Dataframe untuk Visualisasi
daily_orders_df = create_daily_orders_df(main_df)
product_perf_df = create_product_perf_df(main_df)
state_df, city_df = create_geography_df(main_df)
rfm_df = create_rfm_df(main_df)

# --- DASHBOARD LAYOUT ---
st.title('📊 E-Commerce Business Intelligence')
st.markdown(f"Menunjukkan data dari **{start_date}** sehingga **{end_date}**")

# Ringkasan Metrik Utama
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Jumlah Pesanan", value=main_df.order_id.nunique())
with col2:
    total_rev = format_currency(main_df.price.sum(), "BRL", locale='es_CO')
    st.metric("Jumlah Hasil (Revenue)", value=total_rev)
with col3:
    st.metric("Rata-Rata Skor Ulasan", value=round(main_df.review_score.mean(), 2))

st.divider()

# Penggunaan Tab untuk Navigasi UI yang Kemas
tab1, tab2, tab3 = st.tabs(["📦 Ranking Produk", "🌍 Sebaran Geografi", "💎 Analisis Pelanggan (RFM)"])

# TAB 1: PRESTASI PRODUK
with tab1:
    st.subheader("Analisis Hasil & Kepuasan Produk")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Top 10 Kategori Berdasarkan Hasil (Revenue)**")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="price", y="product_category_name", data=product_perf_df.head(10), palette="Blues_d")
        ax.set_xlabel("Total Revenue (BRL)")
        ax.set_ylabel(None)
        st.pyplot(fig)
        
    with col_b:
        st.write("**Top 10 Kategori Berdasarkan Rating Tertinggi (Min. 50 Pesanan)**")
        # Filter kategori dengan pesanan yang mencukupi untuk rating yang adil
        rating_df = product_perf_df[product_perf_df["order_id"] >= 50].sort_values(by="review_score", ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="review_score", y="product_category_name", data=rating_df, palette="Greens_d")
        ax.set_xlim(4, 5) # Fokus pada skor tinggi
        ax.set_xlabel("Average Review Score")
        ax.set_ylabel(None)
        st.pyplot(fig)

    st.info(f"**Insight:** Kategori **{product_perf_df.iloc[0]['product_category_name']}** menjana hasil tertinggi, manakala kategori **{rating_df.iloc[0]['product_category_name']}** mempunyai tahap kepuasan pelanggan yang paling konsisten.")

# TAB 2: TABURAN GEOGRAFI
with tab2:
    st.subheader("Lokasi Pelanggan Utama")
    
    col_c, col_d = st.columns([1, 1.5])
    with col_c:
        st.write("**Pesanan mengikut Negeri (State)**")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(x="order_id", y="customer_state_x", data=state_df, palette="coolwarm")
        ax.set_xlabel("Jumlah Pesanan")
        st.pyplot(fig)
        
    with col_d:
        st.write("**Top 15 Kota dengan Pesanan Terbanyak**")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(x="order_id", y="customer_city_x", data=city_dist_df if 'city_dist_df' in locals() else city_df, palette="magma")
        ax.set_xlabel("Jumlah Pesanan")
        st.pyplot(fig)

    st.warning(f"**Insight:** Kawasan **{state_df.iloc[0]['customer_state_x']}** (Negeri) dan **{city_df.iloc[0]['customer_city_x']}** (Kota) mendominasi jumlah pesanan. Ini adalah pusat logistik utama syarikat.")

# TAB 3: ANALISIS RFM
with tab3:
    st.subheader("Segmentasi Pelanggan (RFM)")
    
    col_e, col_f, col_g = st.columns(3)
    with col_e:
        st.metric("Rata-rata Recency", value=f"{round(rfm_df.recency.mean(), 1)} Hari")
    with col_f:
        st.metric("Rata-rata Frequency", value=f"{round(rfm_df.frequency.mean(), 2)} Pesanan")
    with col_g:
        st.metric("Rata-rata Monetary", value=format_currency(rfm_df.monetary.mean(), "BRL", locale='es_CO'))

    st.write("**Top Pelanggan Berdasarkan Parameter RFM**")
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(35, 15))
    
    # Recency
    sns.barplot(y="recency", x="customer_id", data=rfm_df.sort_values(by="recency", ascending=True).head(5), palette="Blues", ax=ax[0])
    ax[0].set_title("By Recency (Hari)", loc="center", fontsize=50)
    ax[0].set_xticks([])
    
    # Frequency
    sns.barplot(y="frequency", x="customer_id", data=rfm_df.sort_values(by="frequency", ascending=False).head(5), palette="Greens", ax=ax[1])
    ax[1].set_title("By Frequency", loc="center", fontsize=50)
    ax[1].set_xticks([])
    
    # Monetary
    sns.barplot(y="monetary", x="customer_id", data=rfm_df.sort_values(by="monetary", ascending=False).head(5), palette="Oranges", ax=ax[2])
    ax[2].set_title("By Monetary", loc="center", fontsize=50)
    ax[2].set_xticks([])
    
    st.pyplot(fig)

    st.success("**Insight:** Kebanyakan pelanggan adalah pembeli satu kali (*Frequency* rendah). Syarikat perlu fokus pada program kesetiaan untuk meningkatkan pembelian berulang.")

st.divider()
st.caption('Dicoding Data Analysis Project - Randra Ferdian 2024')