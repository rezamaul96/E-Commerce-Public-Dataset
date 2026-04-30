import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import streamlit as st
import os

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Dashboard",
    page_icon="🛒",
    layout="wide"
)

# ─── LOAD DATA ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)

@st.cache_data
def load_all_data():
    orders_df         = pd.read_csv(os.path.join(BASE_DIR, 'orders_dataset.csv'))
    order_payments_df = pd.read_csv(os.path.join(BASE_DIR, 'order_payments_dataset.csv'))
    order_items_df    = pd.read_csv(os.path.join(BASE_DIR, 'order_items_dataset.csv'))
    order_reviews_df  = pd.read_csv(os.path.join(BASE_DIR, 'order_reviews_dataset.csv'))
    products_df       = pd.read_csv(os.path.join(BASE_DIR, 'products_dataset.csv'))
    category_trans_df = pd.read_csv(os.path.join(BASE_DIR, 'product_category_name_translation.csv'))
    return orders_df, order_payments_df, order_items_df, order_reviews_df, products_df, category_trans_df

orders_df, order_payments_df, order_items_df, order_reviews_df, products_df, category_trans_df = load_all_data()

# ─── CLEANING ────────────────────────────────────────────────────────────────
@st.cache_data
def clean_data(orders_df, order_payments_df, order_items_df, order_reviews_df, products_df, category_trans_df):
    # Mengubah tipe data yang kurang benar yaitu datetime pada orders_df
    datetime_cols_orders = [
        'order_purchase_timestamp', 'order_approved_at',
        'order_delivered_carrier_date', 'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]
    for col in datetime_cols_orders:
        orders_df[col] = pd.to_datetime(orders_df[col], errors='coerce')

    # Mengubah tipe data yang kurang benar yaitu datetime pada order_items_df
    order_items_df['shipping_limit_date'] = pd.to_datetime(order_items_df['shipping_limit_date'], errors='coerce')

    # Mengubah tipe data yang kurang benar yaitu datetime pada order_reviews_df
    for col in ['review_creation_date', 'review_answer_timestamp']:
        order_reviews_df[col] = pd.to_datetime(order_reviews_df[col], errors='coerce')

    # 1. Mengisi missing value pada product_df kolom product_category_name dengan 'unknown'
    products_df = products_df.copy()
    products_df['product_category_name'] = products_df['product_category_name'].fillna('unknown')

    # 2. Mengisi missing value kolom dimensi serta deskripsi produk dengan median
    cols_median = [
        'product_name_lenght', 'product_description_lenght', 'product_photos_qty',
        'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm'
    ]
    for col in cols_median:
        if col in products_df.columns:
            products_df[col] = products_df[col].fillna(products_df[col].median())

    # 3. Melakukan filtering data hanya pada status delivered dan menambahkan kolom turunan
    orders_clean = orders_df[orders_df['order_status'] == 'delivered'].copy()
    orders_clean['order_year']       = orders_clean['order_purchase_timestamp'].dt.year
    orders_clean['order_month']      = orders_clean['order_purchase_timestamp'].dt.month
    orders_clean['order_month_name'] = orders_clean['order_purchase_timestamp'].dt.strftime('%b')

    return orders_clean, order_payments_df, order_items_df, order_reviews_df, products_df, category_trans_df

orders_clean, order_payments_df, order_items_df, order_reviews_df, products_df, category_trans_df = \
    clean_data(orders_df, order_payments_df, order_items_df, order_reviews_df, products_df, category_trans_df)

# ─── SIDEBAR: FILTER TANGGAL ─────────────────────────────────────────────────
with st.sidebar:
    st.header("🗓️ Filter Tanggal")
    min_date = orders_clean['order_purchase_timestamp'].dt.date.min()
    max_date = orders_clean['order_purchase_timestamp'].dt.date.max()
    start_date, end_date = st.date_input(
        "Rentang Tanggal",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

# Filter orders_clean berdasarkan tanggal
orders_filtered = orders_clean[
    (orders_clean['order_purchase_timestamp'].dt.date >= start_date) &
    (orders_clean['order_purchase_timestamp'].dt.date <= end_date)
].copy()

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.title("🛒 Dashboard Analisis E-Commerce Public Dataset")
st.markdown("**Nama:** Reza Maulana | **ID:** CDCC299D6Y2532")
st.divider()

# ─── COMPUTE Q1 ──────────────────────────────────────────────────────────────
# Melakukan agregasi payment per order
payment_agg = (
    order_payments_df
    .groupby('order_id')['payment_value']
    .sum()
    .reset_index()
)

# Menggabungkan orders 2018 dengan payment
orders_2018 = orders_filtered[orders_filtered['order_year'] == 2018]
df_q1 = orders_2018.merge(payment_agg, on='order_id', how='left')

# Menghitung revenue per bulannya
monthly_rev = (
    df_q1
    .groupby(['order_month', 'order_month_name'])['payment_value']
    .sum()
    .reset_index()
    .sort_values('order_month')
    .reset_index(drop=True)
)

# ─── COMPUTE Q2 ──────────────────────────────────────────────────────────────
# Mengambil order pada tahun 2017
orders_2017 = orders_filtered[orders_filtered['order_year'] == 2017][['order_id']]

# Melakukan agregasi review score per order (menggunakan median jika terdapat lebih dari 1 review)
reviews_per_order = (
    order_reviews_df
    .groupby('order_id')['review_score']
    .median()
    .reset_index()
)

# Menggabungkan orders 2017, order_items, products, category_trans, reviews
df_q2 = (
    orders_2017
    .merge(order_items_df[['order_id', 'product_id']], on='order_id', how='left')
    .merge(products_df[['product_id', 'product_category_name']], on='product_id', how='left')
    .merge(category_trans_df, on='product_category_name', how='left')
    .merge(reviews_per_order, on='order_id', how='left')
    .dropna(subset=['review_score', 'product_category_name_english'])
)

# Menghitung rata-rata score per kategori
category_score = (
    df_q2
    .groupby('product_category_name_english')
    .agg(
        avg_score    = ('review_score', 'mean'),
        total_orders = ('order_id', 'count')
    )
    .reset_index()
)

# Melakukan filtering dengan minimal 10 order saja supaya data menjadi representif
category_score = category_score[category_score['total_orders'] >= 10]

# Memfilter kategori dengan rata-rata score < 4.0
category_low = (
    category_score[category_score['avg_score'] < 4.0]
    .sort_values('avg_score')
    .reset_index(drop=True)
)

# ─── COMPUTE ANALISIS LANJUTAN YoY ───────────────────────────────────────────
orders_2017_rev = orders_filtered[orders_filtered['order_year'] == 2017]
df_2017_rev     = orders_2017_rev.merge(payment_agg, on='order_id', how='left')

monthly_2017 = (
    df_2017_rev
    .groupby(['order_month', 'order_month_name'])['payment_value']
    .sum().reset_index().sort_values('order_month')
)
monthly_2018 = monthly_rev.copy()

common_months = range(1, 9)
rev_2017 = monthly_2017[monthly_2017['order_month'].isin(common_months)]
rev_2018 = monthly_2018[monthly_2018['order_month'].isin(common_months)]

# ─── COMPUTE RFM ─────────────────────────────────────────────────────────────
# Agregasi payment per order
payment_agg_rfm = (
    order_payments_df
    .groupby('order_id')['payment_value']
    .sum()
    .reset_index()
)

# Gabungkan orders dengan payment
df_rfm = orders_filtered.merge(payment_agg_rfm, on='order_id', how='left')

# Hitung RFM per customer
snapshot_date = df_rfm['order_purchase_timestamp'].dt.date.max()

rfm_df = (
    df_rfm.groupby('customer_id')
    .agg(
        last_order = ('order_purchase_timestamp', 'max'),
        frequency  = ('order_id', 'nunique'),
        monetary   = ('payment_value', 'sum')
    )
    .reset_index()
)

rfm_df['recency'] = rfm_df['last_order'].apply(lambda x: (snapshot_date - x.date()).days)
rfm_df = rfm_df.drop(columns='last_order')

# Scoring RFM menggunakan binning (quintile 1-5)
# Recency: semakin kecil nilainya semakin bagus → skor dibalik
rfm_df['R_score'] = pd.qcut(rfm_df['recency'], q=5, labels=[5, 4, 3, 2, 1]).astype(int)
rfm_df['F_score'] = pd.qcut(rfm_df['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm_df['M_score'] = pd.qcut(rfm_df['monetary'], q=5, labels=[1, 2, 3, 4, 5]).astype(int)

rfm_df['RFM_score'] = rfm_df['R_score'] + rfm_df['F_score'] + rfm_df['M_score']

# Segmentasi pelanggan berdasarkan RFM Score (Manual Grouping)
def segment_customer(score):
    if score >= 13:
        return 'Champions'
    elif score >= 10:
        return 'Loyal Customers'
    elif score >= 7:
        return 'Potential Loyalist'
    elif score >= 5:
        return 'At Risk'
    else:
        return 'Lost Customers'

rfm_df['segment'] = rfm_df['RFM_score'].apply(segment_customer)

segment_summary = (
    rfm_df.groupby('segment')
    .agg(
        jumlah_customer = ('customer_id', 'count'),
        avg_recency     = ('recency', 'mean'),
        avg_frequency   = ('frequency', 'mean'),
        avg_monetary    = ('monetary', 'mean')
    )
    .round(2)
    .reset_index()
    .sort_values('jumlah_customer', ascending=False)
)

# ─── COMPUTE CLUSTERING ───────────────────────────────────────────────────────
# Ambil semua kategori (tidak dibatasi tahun 2017) untuk clustering yang lebih representatif
payment_agg_clust = (
    order_payments_df
    .groupby('order_id')['payment_value']
    .sum()
    .reset_index()
)

df_clust = (
    orders_filtered[['order_id']]
    .merge(order_items_df[['order_id', 'product_id']], on='order_id', how='left')
    .merge(products_df[['product_id', 'product_category_name']], on='product_id', how='left')
    .merge(category_trans_df, on='product_category_name', how='left')
    .merge(order_reviews_df.groupby('order_id')['review_score'].median().reset_index(), on='order_id', how='left')
    .merge(payment_agg_clust, on='order_id', how='left')
    .dropna(subset=['product_category_name_english', 'review_score'])
)

category_perf = (
    df_clust.groupby('product_category_name_english')
    .agg(
        total_orders  = ('order_id', 'count'),
        avg_score     = ('review_score', 'mean'),
        total_revenue = ('payment_value', 'sum')
    )
    .reset_index()
)

# Filter min 50 order agar representatif
category_perf = category_perf[category_perf['total_orders'] >= 50].copy()

# Binning volume order → 3 tier
category_perf['volume_tier'] = pd.cut(
    category_perf['total_orders'],
    bins=[0, 500, 2000, category_perf['total_orders'].max() + 1],
    labels=['Low Volume', 'Mid Volume', 'High Volume']
)

# Binning review score → 3 tier
category_perf['quality_tier'] = pd.cut(
    category_perf['avg_score'],
    bins=[0, 3.8, 4.2, 5.0],
    labels=['Low Quality', 'Mid Quality', 'High Quality']
)

# Cluster label gabungan
category_perf['cluster'] = category_perf['volume_tier'].astype(str) + ' / ' + category_perf['quality_tier'].astype(str)

# ─── SUMMARY METRICS ─────────────────────────────────────────────────────────
if not monthly_rev.empty:
    total   = monthly_rev['payment_value'].sum()
    max_row = monthly_rev.loc[monthly_rev['payment_value'].idxmax()]
    min_row = monthly_rev.loc[monthly_rev['payment_value'].idxmin()]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Total Revenue 2018",       f"R${total/1e6:.2f}M")
    col2.metric("📈 Bulan Tertinggi",           f"{max_row['order_month_name']} — R${max_row['payment_value']/1e6:.2f}M")
    col3.metric("📉 Bulan Terendah",            f"{min_row['order_month_name']} — R${min_row['payment_value']/1e6:.2f}M")
    col4.metric("⚠️ Kategori Bermasalah 2017",  f"{len(category_low)} kategori")
else:
    st.warning("⚠️ Tidak ada data 2018 pada rentang tanggal yang dipilih.")

st.divider()

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Q1 · Revenue Bulanan 2018",
    "⭐ Q2 · Kategori Review Rendah 2017",
    "📈 Analisis Lanjutan · YoY",
    "👥 RFM Analysis",
    "🗂️ Clustering Kategori"
])

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Pertanyaan 1: Berapa total pendapatan penjualan (revenue) yang dihasilkan setiap bulan "
                 "dalam rentang waktu tahun 2018 dalam melihat pola musim penjualan?")

    if monthly_rev.empty:
        st.warning("Tidak ada data 2018 pada rentang tanggal yang dipilih.")
    else:
        labels = monthly_rev['order_month_name'].tolist()
        values = monthly_rev['payment_value'].tolist()
        max_i  = monthly_rev['payment_value'].idxmax()
        min_i  = monthly_rev['payment_value'].idxmin()

        bar_colors = [
            '#e74c3c' if i == max_i else
            '#95a5a6' if i == min_i else
            '#3498db'
            for i in range(len(labels))
        ]

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # --- Bar chart ---
        bars = axes[0].bar(labels, values, color=bar_colors, edgecolor='white', width=0.7)
        for bar, val in zip(bars, values):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 8000,
                f'R${val/1e6:.2f}M',
                ha='center', va='bottom', fontsize=8.5, fontweight='bold'
            )
        axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'R${x/1e6:.1f}M'))
        axes[0].set_ylim(0, max(values) * 1.22)
        axes[0].set_title('Total Revenue per Bulan — 2018', fontsize=13, fontweight='bold', pad=15)
        axes[0].set_xlabel('Bulan', fontsize=11)
        axes[0].set_ylabel('Total Revenue (BRL)', fontsize=11)
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        legend_el = [
            mpatches.Patch(color='#e74c3c', label=f'Tertinggi'),
            mpatches.Patch(color='#95a5a6', label=f'Terendah'),
            mpatches.Patch(color='#3498db', label='Bulan Lainnya'),
        ]
        axes[0].legend(handles=legend_el, fontsize=9)

        # --- Line chart tren ---
        axes[1].plot(
            range(len(labels)), values,
            color='#2ecc71', linewidth=2.5,
            marker='o', markersize=8,
            markerfacecolor='white', markeredgewidth=2.5
        )
        axes[1].fill_between(range(len(labels)), values, alpha=0.12, color='#2ecc71')
        axes[1].axhline(
            y=np.mean(values), color='#e67e22',
            linestyle='--', linewidth=1.5,
            label=f"Rata-rata: R${np.mean(values)/1e6:.2f}M"
        )
        axes[1].annotate(
            f'Puncak\nR${values[max_i]/1e6:.2f}M',
            xy=(max_i, values[max_i]),
            xytext=(max_i - 1.8, values[max_i] - 80000),
            fontsize=9, color='#e74c3c', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5)
        )
        axes[1].set_xticks(range(len(labels)))
        axes[1].set_xticklabels(labels)
        axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'R${x/1e6:.1f}M'))
        axes[1].set_title('Tren Revenue Bulanan 2018 (Pola Musiman)', fontsize=13, fontweight='bold', pad=15)
        axes[1].set_xlabel('Bulan', fontsize=11)
        axes[1].set_ylabel('Total Revenue (BRL)', fontsize=11)
        axes[1].legend(fontsize=9)
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader("📋 Tabel Revenue Bulanan 2018")
        tbl = monthly_rev[['order_month_name', 'payment_value']].copy()
        tbl['% dari Total'] = (tbl['payment_value'] / total * 100).round(1)
        tbl.columns = ['Bulan', 'Total Revenue (R$)', '% dari Total']
        tbl['Total Revenue (R$)'] = tbl['Total Revenue (R$)'].map('{:,.2f}'.format)
        st.dataframe(tbl, use_container_width=True, hide_index=True)

# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Pertanyaan 2: Kategori produk apa yang memiliki rata-rata skor di bawah 4.0 pada tahun 2017 "
                 "sebagai identifikasi produk yang perlu ditingkatkan kualitasnya?")

    if category_low.empty:
        st.warning("Tidak ada data kategori bermasalah pada rentang tanggal yang dipilih.")
    else:
        bar_colors2 = [
            '#e74c3c' if s < 3.75 else '#e67e22'
            for s in category_low['avg_score']
        ]

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # --- Horizontal bar chart ---
        bars2 = axes[0].barh(
            category_low['product_category_name_english'],
            category_low['avg_score'],
            color=bar_colors2, edgecolor='white', height=0.65
        )
        for bar, val in zip(bars2, category_low['avg_score']):
            axes[0].text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.2f} ⭐',
                va='center', ha='left', fontsize=9
            )
        axes[0].axvline(x=4.0, color='gray', linestyle='--', linewidth=1.5, label='Threshold 4.0')
        axes[0].axvline(
            x=category_score['avg_score'].mean(), color='#3498db',
            linestyle=':', linewidth=1.5,
            label=f"Rata-rata semua kategori: {category_score['avg_score'].mean():.2f}"
        )
        axes[0].set_xlim(3.0, 4.4)
        axes[0].set_title('Kategori dengan Avg Review Score < 4.0\nTahun 2017', fontsize=13, fontweight='bold', pad=15)
        axes[0].set_xlabel('Rata-rata Review Score', fontsize=11)
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        legend_el2 = [
            mpatches.Patch(color='#e74c3c', label='Score < 3.75 (Kritis)'),
            mpatches.Patch(color='#e67e22', label='Score 3.75–4.0 (Perlu perbaikan)'),
            plt.Line2D([0],[0], color='gray', linestyle='--', label='Threshold 4.0'),
            plt.Line2D([0],[0], color='#3498db', linestyle=':', label=f"Rata-rata: {category_score['avg_score'].mean():.2f}"),
        ]
        axes[0].legend(handles=legend_el2, fontsize=8, loc='lower right')

        # --- Bubble chart: score vs volume order ---
        scatter = axes[1].scatter(
            category_low['total_orders'],
            category_low['avg_score'],
            s=category_low['total_orders'] / 1.5,
            c=category_low['avg_score'],
            cmap='RdYlGn', vmin=3.5, vmax=4.0,
            alpha=0.85, edgecolors='white', linewidth=1.2
        )
        for _, row in category_low.iterrows():
            if row['total_orders'] > 150 or row['avg_score'] < 3.7:
                axes[1].annotate(
                    row['product_category_name_english'].replace('_', ' '),
                    xy=(row['total_orders'], row['avg_score']),
                    xytext=(8, 4), textcoords='offset points',
                    fontsize=8, color='#2c3e50'
                )
        axes[1].axhline(y=4.0, color='gray', linestyle='--', linewidth=1.5, label='Threshold 4.0')
        plt.colorbar(scatter, ax=axes[1], label='Avg Review Score')
        axes[1].set_title('Volume Order vs Avg Review Score\nKategori Bermasalah (2017)', fontsize=13, fontweight='bold', pad=15)
        axes[1].set_xlabel('Total Order (2017)', fontsize=11)
        axes[1].set_ylabel('Rata-rata Review Score', fontsize=11)
        axes[1].legend(fontsize=9)
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader("📋 Tabel Kategori Bermasalah 2017")
        tbl2 = category_low[['product_category_name_english', 'avg_score', 'total_orders']].copy()
        tbl2.columns = ['Kategori', 'Avg Review Score', 'Total Order']
        tbl2['Avg Review Score'] = tbl2['Avg Review Score'].round(3)
        tbl2['Status'] = tbl2['Avg Review Score'].apply(lambda x: '🔴 Kritis' if x < 3.75 else '🟠 Perlu Perbaikan')
        st.dataframe(tbl2, use_container_width=True, hide_index=True)

# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Analisis Lanjutan: Perbandingan Revenue 2017 vs 2018 (Jan–Agu)")

    if rev_2017.empty or rev_2018.empty:
        st.warning("Data tidak cukup untuk perbandingan YoY pada rentang tanggal yang dipilih.")
    else:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(range(len(rev_2017)), rev_2017['payment_value'],
                color='#95a5a6', linewidth=2, marker='o', markersize=6,
                markerfacecolor='white', markeredgewidth=2, label='2017')
        ax.plot(range(len(rev_2018)), rev_2018['payment_value'],
                color='#3498db', linewidth=2.5, marker='o', markersize=6,
                markerfacecolor='white', markeredgewidth=2, label='2018')

        ax.set_xticks(range(len(rev_2017)))
        ax.set_xticklabels(rev_2017['order_month_name'].tolist())
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'R${x/1e6:.1f}M'))
        ax.set_title('Perbandingan Revenue Bulanan: 2017 vs 2018 (Jan–Agu)', fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel('Bulan', fontsize=11)
        ax.set_ylabel('Total Revenue (BRL)', fontsize=11)
        ax.legend(fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.info(
            "💡 **Insight:**\n\n"
            "- Revenue 2018 konsisten **lebih tinggi** dari 2017 di semua bulan Jan–Agu\n"
            "- Menandakan **pertumbuhan bisnis YoY yang sehat**\n"
            "- Pola musiman keduanya mirip — naik di awal tahun lalu stabil\n"
            "- Ini mengkonfirmasi adanya **seasonality** yang bisa dimanfaatkan untuk perencanaan kampanye di tahun berikutnya"
        )

        merged_yoy = rev_2017[['order_month', 'order_month_name', 'payment_value']].merge(
            rev_2018[['order_month', 'payment_value']], on='order_month', suffixes=('_2017', '_2018')
        )
        merged_yoy['YoY Growth (%)'] = (
            (merged_yoy['payment_value_2018'] - merged_yoy['payment_value_2017'])
            / merged_yoy['payment_value_2017'] * 100
        ).round(1)
        merged_yoy = merged_yoy.rename(columns={
            'order_month_name': 'Bulan',
            'payment_value_2017': 'Revenue 2017 (R$)',
            'payment_value_2018': 'Revenue 2018 (R$)'
        })
        merged_yoy['Revenue 2017 (R$)'] = merged_yoy['Revenue 2017 (R$)'].map('{:,.0f}'.format)
        merged_yoy['Revenue 2018 (R$)'] = merged_yoy['Revenue 2018 (R$)'].map('{:,.0f}'.format)

        st.subheader("📋 Tabel YoY Growth per Bulan")
        st.dataframe(
            merged_yoy[['Bulan', 'Revenue 2017 (R$)', 'Revenue 2018 (R$)', 'YoY Growth (%)']],
            use_container_width=True, hide_index=True
        )

# ── TAB 4: RFM ────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("RFM Analysis")
    st.markdown(
        "RFM Analysis digunakan untuk mengelompokkan pelanggan berdasarkan perilaku pembelian mereka:\n"
        "- **Recency (R):** Berapa hari sejak terakhir kali pelanggan melakukan pembelian\n"
        "- **Frequency (F):** Berapa kali pelanggan melakukan transaksi\n"
        "- **Monetary (M):** Berapa total pengeluaran pelanggan\n\n"
        "Dataset yang digunakan: `orders_clean` + `order_payments_df`"
    )

    if rfm_df.empty:
        st.warning("Data tidak cukup untuk RFM Analysis pada rentang tanggal yang dipilih.")
    else:
        segment_order = ['Champions', 'Loyal Customers', 'Potential Loyalist', 'At Risk', 'Lost Customers']
        colors_seg    = ['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c']
        seg_count     = rfm_df['segment'].value_counts().reindex(segment_order).fillna(0)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # --- Bar: Jumlah Pelanggan per Segmen ---
        bars = axes[0].bar(segment_order, seg_count.values, color=colors_seg, edgecolor='white', width=0.6)
        for bar, val in zip(bars, seg_count.values):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 50,
                f'{int(val):,}', ha='center', va='bottom', fontsize=9, fontweight='bold'
            )
        axes[0].set_title('Jumlah Pelanggan per Segmen RFM', fontsize=12, fontweight='bold', pad=12)
        axes[0].set_xlabel('Segmen', fontsize=10)
        axes[0].set_ylabel('Jumlah Pelanggan', fontsize=10)
        axes[0].tick_params(axis='x', rotation=20)
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)

        # --- Bar: Avg Monetary per Segmen ---
        avg_mon = rfm_df.groupby('segment')['monetary'].mean().reindex(segment_order)
        bars2   = axes[1].bar(segment_order, avg_mon.values, color=colors_seg, edgecolor='white', width=0.6)
        for bar, val in zip(bars2, avg_mon.values):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                f'R${val:,.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold'
            )
        axes[1].set_title('Rata-rata Monetary per Segmen', fontsize=12, fontweight='bold', pad=12)
        axes[1].set_xlabel('Segmen', fontsize=10)
        axes[1].set_ylabel('Avg Monetary (R$)', fontsize=10)
        axes[1].tick_params(axis='x', rotation=20)
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)

        # --- Bar: Avg Recency per Segmen ---
        avg_rec = rfm_df.groupby('segment')['recency'].mean().reindex(segment_order)
        bars3   = axes[2].bar(segment_order, avg_rec.values, color=colors_seg, edgecolor='white', width=0.6)
        for bar, val in zip(bars3, avg_rec.values):
            axes[2].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f'{val:.0f}d', ha='center', va='bottom', fontsize=9, fontweight='bold'
            )
        axes[2].set_title('Rata-rata Recency per Segmen (hari)', fontsize=12, fontweight='bold', pad=12)
        axes[2].set_xlabel('Segmen', fontsize=10)
        axes[2].set_ylabel('Avg Recency (hari)', fontsize=10)
        axes[2].tick_params(axis='x', rotation=20)
        axes[2].spines['top'].set_visible(False)
        axes[2].spines['right'].set_visible(False)

        plt.suptitle('RFM Analysis — Segmentasi Pelanggan E-Commerce', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader("📋 Tabel Segmentasi Pelanggan")
        st.dataframe(segment_summary, use_container_width=True, hide_index=True)

# ── TAB 5: CLUSTERING ─────────────────────────────────────────────────────────
with tab5:
    st.subheader("Clustering: Segmentasi Kategori Produk Berdasarkan Performa")
    st.markdown(
        "Clustering manual menggunakan **binning** untuk mengelompokkan kategori produk berdasarkan dua dimensi:\n"
        "- **Volume Order:** Seberapa banyak produk terjual\n"
        "- **Avg Review Score:** Seberapa puas pelanggan\n\n"
        "Hasilnya digunakan untuk menentukan prioritas tindakan bisnis."
    )

    if category_perf.empty:
        st.warning("Data tidak cukup untuk Clustering pada rentang tanggal yang dipilih.")
    else:
        cluster_colors = {
            'High Volume / High Quality'  : '#2ecc71',
            'High Volume / Mid Quality'   : '#f39c12',
            'High Volume / Low Quality'   : '#e74c3c',
            'Mid Volume / High Quality'   : '#3498db',
            'Mid Volume / Mid Quality'    : '#95a5a6',
            'Mid Volume / Low Quality'    : '#e67e22',
            'Low Volume / High Quality'   : '#1abc9c',
            'Low Volume / Mid Quality'    : '#bdc3c7',
            'Low Volume / Low Quality'    : '#c0392b',
        }

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # --- Scatter: Volume vs Score per cluster ---
        for cluster_name, group in category_perf.groupby('cluster'):
            color = cluster_colors.get(cluster_name, '#888888')
            axes[0].scatter(
                group['total_orders'], group['avg_score'],
                s=group['total_revenue'] / 5000,
                color=color, alpha=0.75, edgecolors='white', linewidth=0.8,
                label=cluster_name
            )
            for _, row in group.iterrows():
                if row['total_orders'] > 5000 or row['avg_score'] < 3.8:
                    axes[0].annotate(
                        row['product_category_name_english'].replace('_', ' '),
                        xy=(row['total_orders'], row['avg_score']),
                        xytext=(6, 4), textcoords='offset points',
                        fontsize=7, color='#2c3e50'
                    )

        axes[0].axhline(y=4.2, color='#2ecc71', linestyle='--', linewidth=1.2, alpha=0.6, label='High Quality threshold (4.2)')
        axes[0].axhline(y=3.8, color='#e74c3c', linestyle='--', linewidth=1.2, alpha=0.6, label='Low Quality threshold (3.8)')
        axes[0].set_title('Clustering Kategori Produk\n(Volume Order vs Avg Review Score)', fontsize=12, fontweight='bold', pad=12)
        axes[0].set_xlabel('Total Order', fontsize=10)
        axes[0].set_ylabel('Avg Review Score', fontsize=10)
        axes[0].legend(fontsize=7, loc='lower right', ncol=1)
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)

        # --- Bar: Jumlah Kategori per Cluster ---
        cluster_count = category_perf['cluster'].value_counts().sort_values(ascending=True)
        bar_colors_c  = [cluster_colors.get(c, '#888') for c in cluster_count.index]
        axes[1].barh(cluster_count.index, cluster_count.values, color=bar_colors_c, edgecolor='white', height=0.65)
        for i, val in enumerate(cluster_count.values):
            axes[1].text(val + 0.2, i, str(val), va='center', fontsize=9, fontweight='bold')
        axes[1].set_title('Jumlah Kategori per Cluster', fontsize=12, fontweight='bold', pad=12)
        axes[1].set_xlabel('Jumlah Kategori', fontsize=10)
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)

        plt.suptitle('Clustering Kategori Produk Berdasarkan Volume & Kualitas', fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader("📋 Tabel Cluster Kategori Produk")
        tbl_clust = category_perf[['product_category_name_english', 'total_orders', 'avg_score', 'cluster']].copy()
        tbl_clust.columns = ['Kategori', 'Total Order', 'Avg Score', 'Cluster']
        tbl_clust['Avg Score'] = tbl_clust['Avg Score'].round(3)
        st.dataframe(tbl_clust.sort_values('Total Order', ascending=False), use_container_width=True, hide_index=True)

# ─── CONCLUSION ──────────────────────────────────────────────────────────────
st.divider()
st.caption("<p style='text-align: center;'>Copyright (c) Reza Maulana 2026</p>", unsafe_allow_html=True)
