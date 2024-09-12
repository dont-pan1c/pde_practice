import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt



def pde_sol(F, U, t0, t_end, X_0, params):
    t = sp.symbols('t')
    X = sp.IndexedBase('X')
    # Алгортим расчёта

    # 1. получение полной системы уравнений в символьном виде
    E = np.add(F, U)

    # 2. преобразуем символьные уравнения в функции для численных расчётов
    E_func = []
    for equation in E:
        E_func.append(sp.lambdify((t, X), equation.subs(params), "numpy"))

    # 3. вспомогательная функция для метода Рунге-Кутта
    system_pde = lambda t, z : [func(t, z) for func in E_func]

    # 4. вызов метода Рунге-Кутта
    sol = solve_ivp(system_pde, [t0, t_end], X_0, method='RK45', t_eval=np.linspace(t0, t_end, 1000))

    # 5. визуализация результатов
    for i, y in enumerate(sol.y):
        plt.plot(sol.t, y, label='Популяция X_{}'.format(i))


    plt.title('Метод Рунге-Кутта')
    plt.xlabel('Время t')
    plt.ylabel('Популяции')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
