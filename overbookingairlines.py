# app.py - Interactive Airline Overbooking Optimizer (Streamlit + Numba + Joblib)
import streamlit as st
import numpy as np
import pandas as pd
import time
import itertools
import multiprocessing
from numba import jit
import math
from joblib import Parallel, delayed
import plotly.express as px
import plotly.graph_objects as go

# --------------------------- NUMBA FUNCTIONS ---------------------------
@jit(nopython=True)
def log_binomial_pmf(k, n, p):
    if k < 0 or k > n:
        return -np.inf
    if p == 0.0:
        return 0.0 if k == 0 else -np.inf
    if p == 1.0:
        return 0.0 if k == n else -np.inf
    log_p = math.log(p)
    log_one_minus_p = math.log(1.0 - p)
    log_comb = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    return log_comb + k * log_p + (n - k) * log_one_minus_p

@jit(nopython=True)
def calculate_compensation_cost(i, j, k, C_vec, T_prime_vec):
    C_I, C_II, C_III = C_vec
    T_I_prime, T_II_prime, T_III_prime = T_prime_vec

    avail_I = max(0.0, C_I - i)
    excess_I = max(0.0, i - C_I)
    cost_I = T_I_prime * excess_I

    avail_II = max(0.0, C_II - j)
    excess_II = max(0.0, j - C_II)
    net_bumps_II = max(0.0, excess_II - avail_I)
    cost_II = T_II_prime * net_bumps_II

    excess_III = max(0.0, k - C_III)
    avail_I_after_II = max(0.0, avail_I - excess_II)
    total_avail_for_III = avail_II + avail_I_after_II
    net_bumps_III = max(0.0, excess_III - total_avail_for_III)
    cost_III = T_III_prime * net_bumps_III

    return cost_I + cost_II + cost_III

@jit(nopython=True)
def calculate_expected_revenue(B_vec, C_vec, p_vec, T_vec, T_prime_vec, T_fee_vec):
    B_I, B_II, B_III = B_vec
    t_ex_rev = 0.0

    for i in range(B_I + 1):
        log_p_i = log_binomial_pmf(i, B_I, p_vec[0])
        if log_p_i == -np.inf:
            continue

        for j in range(B_II + 1):
            log_p_j = log_binomial_pmf(j, B_II, p_vec[1])
            if log_p_j == -np.inf:
                continue

            log_p_ij = log_p_i + log_p_j

            for k in range(B_III + 1):
                log_p_k = log_binomial_pmf(k, B_III, p_vec[2])
                if log_p_k == -np.inf:
                    continue

                p_ijk = math.exp(log_p_ij + log_p_k)
                if p_ijk <= 0:
                    continue

                rev_show_up = i * T_vec[0] + j * T_vec[1] + k * T_vec[2]
                rev_no_show = (B_I - i) * T_fee_vec[0] + (B_II - j) * T_fee_vec[1] + (B_III - k) * T_fee_vec[2]
                cost_F = calculate_compensation_cost(i, j, k, C_vec, T_prime_vec)
                rev_ijk = rev_show_up + rev_no_show - cost_F

                t_ex_rev += p_ijk * rev_ijk

    return t_ex_rev

# --------------------------- STREAMLIT APP ---------------------------
st.set_page_config(page_title="Airline Overbooking Optimizer", 
                   layout="wide",
                   initial_sidebar_state="expanded")
st.title("Airline Overbooking Revenue Optimizer", )
st.markdown("""
Ready to maximize your airline's revenue, capitalists? 😎
""")
st.markdown(""" Find the exact number of tickets to sell in each class (Business, Premium Economy, Economy) to maximize your profit, predicting no-shows and accounting for the complex costs of bumping passengers.""")

# Sidebar inputs
with st.sidebar:
    st.header("Aircraft & Demand Parameters")
    col1, col2 = st.columns(2)
    with col1:
        C_I = st.number_input("Seats Class I (Premium)", 10, 50, 20)
        C_II = st.number_input("Seats Class II (Business)", 30, 100, 50)
        C_III = st.number_input("Seats Class III (Economy)", 50, 300, 100)
    with col2:
        p_I = st.slider("Show-up Rate Class I", 0.70, 1.00, 0.96, 0.01)
        p_II = st.slider("Show-up Rate Class II", 0.70, 1.00, 0.95, 0.01)
        p_III = st.slider("Show-up Rate Class III", 0.70, 1.00, 0.95, 0.01)

    st.header("Ticket Prices & Penalties")
    col3, col4 = st.columns(2)
    with col3:
        T_I = st.number_input("Ticket Price Class I ($)", 300, 2000, 500)
        T_II = st.number_input("Ticket Price Class II ($)", 200, 1000, 300)
        T_III = st.number_input("Ticket Price Class III ($)", 100, 600, 200)
    with col4:
        k_I = st.number_input("Bump Penalty Multiplier I", 1.0, 5.0, 1.0, 0.1)
        k_II = st.number_input("Bump Penalty Multiplier II", 1.0, 5.0, 1.0, 0.1)
        k_III = st.number_input("Bump Penalty Multiplier III", 1.0, 5.0, 1.0, 0.1)

    st.header("No-Show Refund Policy")
    refund_I = st.slider("Refund % if no-show (Class I)", 0.0, 1.0, 1.00, 0.05)
    refund_II = st.slider("Refund % if no-show (Class II)", 0.0, 1.0, 0.75, 0.05)
    refund_III = st.slider("Refund % if no-show (Class III)", 0.0, 1.0, 0.50, 0.05)

    st.header("Search Settings")
    search_mode = st.radio("Search Mode", ["Fast (Coarse → Fine)", "Brute Force Small Range", "Custom Range"])
    
    if search_mode == "Fast (Coarse → Fine)":
        coarse_step_I = st.slider("Coarse step Class I", 1, 6, 3)
        coarse_step_II = st.slider("Coarse step Class II", 3, 10, 5)
        coarse_step_III = st.slider("Coarse step Class III", 3, 15, 5)
        max_over_I = st.slider("Max overbook Class I", 5, 30, 15)
        max_over_II = st.slider("Max overbook Class II", 10, 60, 25)
        max_over_III = st.slider("Max overbook Class III", 15, 100, 35)
        fine_radius = st.slider("Fine search radius", 1, 5, 2)

# Prepare arrays
C_vec = np.array([C_I, C_II, C_III], dtype=np.int64)
p_vec = np.array([p_I, p_II, p_III], dtype=np.float64)
T_vec = np.array([T_I, T_II, T_III], dtype=np.float64)
k_vec = np.array([k_I, k_II, k_III], dtype=np.float64)
T_prime_vec = T_vec * (1 + k_vec)
T_fee_vec = T_vec * np.array([refund_I, refund_II, refund_III])

# Run button
if st.button("Run Overbooking Optimization", type="primary"):
    with st.spinner("Optimizing... This may take 10–60 seconds depending on grid size..."):
        start_total = time.time()

        # Determine search ranges
        if search_mode == "Fast (Coarse → Fine)":
            coarse_I_range = range(C_I, C_I + max_over_I + 1, coarse_step_I)
            coarse_II_range = range(C_II, C_II + max_over_II + 1, coarse_step_II)
            coarse_III_range = range(C_III, C_III + max_over_III + 1, coarse_step_III)

            # Stage 1: Coarse
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Stage 1/2: Coarse search in progress...")

            grid_coarse = list(itertools.product(coarse_I_range, coarse_II_range, coarse_III_range))
            n_total = len(grid_coarse)

            results_coarse = []
            for idx, B_tuple in enumerate(grid_coarse):
                B_vec_np = np.array(B_tuple, dtype=np.int64)
                rev = calculate_expected_revenue(B_vec_np, C_vec, p_vec, T_vec, T_prime_vec, T_fee_vec)
                results_coarse.append((*B_tuple, rev))
                if idx % max(1, n_total//50) == 0:
                    progress_bar.progress((idx + 1) / n_total)

            df_coarse = pd.DataFrame(results_coarse, columns=["B_I", "B_II", "B_III", "Revenue"])
            best_coarse = df_coarse.loc[df_coarse["Revenue"].idxmax()]

            # Stage 2: Fine
            status_text.text("Stage 2/2: Fine search around best candidate...")
            fine_I = range(max(C_I, int(best_coarse.B_I) - fine_radius), int(best_coarse.B_I) + fine_radius + 1)
            fine_II = range(max(C_II, int(best_coarse.B_II) - fine_radius), int(best_coarse.B_II) + fine_radius + 1)
            fine_III = range(max(C_III, int(best_coarse.B_III) - fine_radius), int(best_coarse.B_III) + fine_radius + 1)

            grid_fine = list(itertools.product(fine_I, fine_II, fine_III))
            results_fine = []
            for idx, B_tuple in enumerate(grid_fine):
                B_vec_np = np.array(B_tuple, dtype=np.int64)
                rev = calculate_expected_revenue(B_vec_np, C_vec, p_vec, T_vec, T_prime_vec, T_fee_vec)
                results_fine.append((*B_tuple, rev))

            df_fine = pd.DataFrame(results_fine, columns=["B_I", "B_II", "B_III", "Revenue"])
            best_final = df_fine.loc[df_fine["Revenue"].idxmax()]

            final_df = df_fine

        else:
            st.error("Other modes coming soon! Use Fast mode for now.")
            st.stop()

        end_total = time.time()
        progress_bar.progress(1.0)
        status_text.text(f"Optimization complete in {end_total - start_total:.1f} seconds!")

    # --------------------------- RESULTS DISPLAY ---------------------------
    st.success("Optimization Complete!")

    colA, colB, colC, colD = st.columns(4)
    with colA:
        st.metric("Optimal Tickets Class I", f"{int(best_final.B_I)}", f"+{int(best_final.B_I - C_I)}")
    with colB:
        st.metric("Optimal Tickets Class II", f"{int(best_final.B_II)}", f"+{int(best_final.B_II - C_II)}")
    with colC:
        st.metric("Optimal Tickets Class III", f"{int(best_final.B_III)}", f"+{int(best_final.B_III - C_III)}")
    with colD:
        st.metric("Expected Revenue", f"${best_final.Revenue:,.0f}")

    st.subheader("Top 10 Strategies")
    top10 = final_df.sort_values("Revenue", ascending=False).head(10).copy()
    top10["Over I"] = top10.B_I - C_I
    top10["Over II"] = top10.B_II - C_II
    top10["Over III"] = top10.B_III - C_III
    st.dataframe(top10[["B_I","Over I","B_II","Over II","B_III","Over III","Revenue"]], use_container_width=True)


    # Revenue vs Overbooking
    fig2 = go.Figure()
    for cls, name, color in zip([0,1,2], ["Class I", "Class II", "Class III"], ["crimson", "goldenrod", "teal"]):
        over = final_df.iloc[:, cls] - C_vec[cls]
        fig2.add_trace(go.Scatter(
            x=over, y=final_df["Revenue"],
            mode="markers", name=name,
            marker=dict(color=color, size=8, opacity=0.6)
        ))
    fig2.update_layout(
        title="Revenue vs Overbooking Level per Class",
        xaxis_title="Overbooking Amount",
        yaxis_title="Expected Revenue ($)",
        legend_title="Fare Class"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Download results
    csv = final_df.to_csv(index=False)
    st.download_button("📥 Download Full Results (CSV)", csv, "overbooking_results.csv", "text/csv")

st.caption("Built by Trương Gia Hân, Đào Thu Huyền, Trần Phương Anh | Fulbright University Vietnam")