import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---------------------------------------------------------
# 1. 시뮬레이션 설정
# ---------------------------------------------------------
# 점의 개수를 리스트로 설정합니다.
n_simulations_list = [1000, 10000, 100000]
np.random.seed(42) # 결과 재현을 위한 시드 고정

# ---------------------------------------------------------
# 2. 그래프 생성 루프
# ---------------------------------------------------------
for N in n_simulations_list:
    # --- 무작위 시행 및 판정 ---
    x = np.random.rand(N)
    y = np.random.rand(N)
    distance_squared = x**2 + y**2
    inside_circle = distance_squared <= 1

    # --- 원주율 근사값 계산 ---
    points_inside = np.sum(inside_circle)
    pi_estimate = 4 * (points_inside / N)

    # --- 시각화 (Plotting) ---
    fig, ax = plt.subplots(figsize=(8, 8))

    # 원 안의 점 (파란색)
    # 점이 많아질수록 겹쳐 보이므로 크기(s)와 투명도(alpha)를 조절합니다.
    point_size = 2 if N <= 10000 else 1
    alpha_val = 0.5 if N <= 10000 else 0.3

    ax.scatter(x[inside_circle], y[inside_circle], color='blue', s=point_size, alpha=alpha_val, label='Inside Circle')
    # 원 밖의 점 (빨간색)
    ax.scatter(x[~inside_circle], y[~inside_circle], color='red', s=point_size, alpha=alpha_val, label='Outside Circle')

    # 사분원 경계선 그리기
    circle_arc = patches.Arc((0, 0), 2, 2, theta1=0, theta2=90, color='black', linewidth=2)
    ax.add_patch(circle_arc)

    # 그래프 꾸미기
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'Monte Carlo Simulation: $\pi$ Approximation (N={N:,})', fontsize=15, fontweight='bold')
    ax.legend(loc='lower left')

    # ★ 결과 텍스트 상자 ★
    text_str = f'Target $\pi$: {np.pi:.6f}\nEstimated $\pi$: {pi_estimate:.6f}\nError: {abs(np.pi - pi_estimate):.6f}'
    ax.text(0.95, 0.95, text_str, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    plt.show() # 그래프 하나 출력 후 닫으면 다음 그래프 생성