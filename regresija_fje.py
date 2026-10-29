import os
import numpy as np
from itertools import combinations_with_replacement
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd


# Generisanje obelezja za polinomijalnu regresiju
def generate_polynomial_features(X: np.ndarray,
                                 p: int,
) -> np.ndarray:
    """
    Generise polinomna obelezja na osnovu matrice obelezja X do stepena p.

    :param X: Ulazna matrica obelezja dimenzija (n,m) - n je broj primera a m broj obelezja
    :param p: Maksimalni stepen polinoma
    :return: Matrica sa svim obelezjima X_out.
    """
    n, m = X.shape
    cols = []

    # Generisem sve kombinacije sa ponavljanjem
    for d in range(1, p + 1):
        for combo in combinations_with_replacement(range(m), d):
            col = np.prod(X[:, combo], axis=1, keepdims=True)
            cols.append(col)

    return np.hstack(cols) if cols else np.empty((n, 0))


def standardize_features(X_train: np.ndarray,
                         X_val: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardizuje ulazne matrice obelezja oduzimanjem srednje vrednosti i deljenjem sa standardnom devijacijom.

    :param X_train: Trening matrica obelezja
    :param X_val: Opciona validaciona/test matrica obelezja
    :return: Skalirani trening, skalirani validacioni (ili None), srednje vrednosti i standardne devijacije
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0.0] = 1.0  # Sprečavanje deljenja sa nulom

    X_train_scaled = (X_train - mean) / std
    X_val_scaled = (X_val - mean) / std if X_val is not None else None

    return X_train_scaled, X_val_scaled, mean, std


def soft_threshold(rho: float,
                   alpha: float,
) -> float:
    """
    Primenjuje soft-thresholding operator za LASSO optimizaciju.

    :param rho: Vrednost korelacije/reziduala
    :param alpha: Regularizacioni parametar (prag)
    :return: Skalirana vrednost nakon pragiranja
    """
    if rho > alpha:
        return rho - alpha
    elif rho < -alpha:
        return rho + alpha
    else:
        return 0.0


def coordinate_descent_lasso(X: np.ndarray,
                             y: np.ndarray,
                             lmbda: float,
                             max_iter: int = 1000,
                             tol: float = 1e-4,
                             return_history: bool = False,
) -> tuple [np.ndarray, float, np.ndarray]:
    """
    Sprovodi LASSO regresiju koriscenjem metode koordinatnog spusta (coordinate descent).

    :param X: Matrica obelezja dimenzija (n, m)
    :param y: Vektor ciljne promenljive dimenzija (n,)
    :param lmbda: Regularizacioni parametar lambda
    :param max_iter: Maksimalan broj iteracija
    :param tol: Tolerancija za uslov zaustavljanja
    :param return_history: Ako je True, vraca i istoriju vrednosti w kroz iteracije
    :return: Vektor koeficijenata w, slobodni clan b (i opciono w_history)
    """
    n, m = X.shape
    w = np.zeros(m)
    b = np.mean(y)

    w_history = [w.copy()] if return_history else None

    for iteration in range(max_iter):
        w_old = w.copy()

        # Ažuriranje slobodnog člana
        b = np.mean(y - np.dot(X, w))

        for i in range(m):
            # Izračunavanje reziduala bez i-tog obeležja
            r_i = y - (b + np.dot(X, w)) + X[:, i] * w[i]

            a_i = np.dot(X[:, i], X[:, i])
            if a_i == 0:
                continue
            c_i = np.dot(X[:, i], r_i)

            # Ažuriranje težine w_i pomoću soft-thresholding funkcije
            w[i] = soft_threshold(c_i, lmbda * n) / a_i

        if return_history:
            w_history.append(w.copy())

        # Uslov konvergencije
        if np.linalg.norm(w - w_old, ord=1) < tol:
            #print(f"Konvergencija, iteracija {iteration}.")
            break

    if return_history:
        return w, b, np.array(w_history)
    return w, b, np.empty(0)


def predict(X: np.ndarray,
            w: np.ndarray,
            b: float,
) -> np.ndarray:
    """
    Izracunava predikciju modela za datu matricu obelezja X.

    :param X: Matrica obelezja dimenzija (n, m)
    :param w: Vektor koeficijenata dimenzija (m,)
    :param b: Slobodni clan
    :return: Vektor predikcija dimenzija (n,)
    """
    return np.dot(X, w) + b


def compute_rmse(y_true: np.ndarray,
                 y_pred: np.ndarray,
) -> float:
    """
    Racuna koren srednje-kvadratne greske (RMSE) izmedju stvarnih i predvidenih vrednosti.

    :param y_true: Stvarne vrednosti
    :param y_pred: Predvidjene vrednosti
    :return: Vrednost RMSE
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def split_fold(X: np.ndarray,
               y: np.ndarray,
               k: int,
               i: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vrsi podelu podataka na trening i validacioni skup za k-fold unakrsnu validaciju.

    :param X: Matrica obelezja
    :param y: Vektor ciljne promenljive
    :param k: Ukupan broj foldova
    :param i: Indeks tekuceg folda (0 do k-1)
    :return: X_train, y_train, X_val, y_val
    """
    n = X.shape[0]
    indices = np.arange(n)
    fold_size = n // k

    start_idx = i * fold_size
    end_idx = (i + 1) * fold_size if i != k - 1 else n

    val_indices = indices[start_idx:end_idx]
    train_indices = np.setdiff1d(indices, val_indices)

    return X[train_indices], y[train_indices], X[val_indices], y[val_indices]


def cross_validation_kfold(X: np.ndarray,
                           y: np.ndarray,
                           k: int,
                           p: int,
                           lmbda: float,
) -> float:
    """
    Izracunava prosecnu RMSE vrednost koriscenjem k-fold unakrsnu validaciju za dati stepen polinoma p i lambda.

    :param X: Ulazna matrica obelezja
    :param y: Ciljna promenljiva
    :param k: Broj foldova
    :param p: Stepen polinoma
    :param lmbda: Regularizacioni parametar
    :return: Prosecna RMSE vrednost na validacionim skupovima
    """
    n = X.shape[0]
    np.random.seed(42)
    perm = np.random.permutation(n)
    X_shuffled = X[perm]
    y_shuffled = y[perm]

    rmses = []

    for i in range(k):
        X_train, y_train, X_val, y_val = split_fold(X_shuffled, y_shuffled, k, i)

        # Generisanje polinomnih obeležja
        X_train_poly = generate_polynomial_features(X_train, p)
        X_val_poly = generate_polynomial_features(X_val, p)

        # Standardizacija obeležja
        X_train_scaled, X_val_scaled, _, _ = standardize_features(X_train_poly, X_val_poly)

        # Obučavanje modela (LASSO)
        w, b, _ = coordinate_descent_lasso(X_train_scaled, y_train, lmbda)

        # Predikcija i evaluacija
        y_pred = predict(X_val_scaled, w, b)
        rmse = compute_rmse(y_val, y_pred)
        rmses.append(rmse)

    return np.mean(rmses)


def _save_plot_if_needed(name: str = None) -> None:
    """
    Pomoćna funkcija koja podešava tajt raspored, lagan grid i čuva sliku u folder 'slike' ako je prosleđeno ime.
    """
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    if name is not None:
        folder_path = "slike"
        os.makedirs(folder_path, exist_ok=True)
        if not name.endswith('.pdf'):
            name += '.pdf'
        file_path = os.path.join(folder_path, name)
        plt.savefig(file_path, format='pdf', dpi=300, bbox_inches='tight')

    plt.show()


def plot_hyperparameter_selection(lambdas: list[float],
                                  degrees: list[int],
                                  rmse_results: np.ndarray,
                                  name: str = None,
) -> None:
    """
    Kreira grafikon koji demonstrira izbor hiperparametara (lambda i stepen polinoma p) na osnovu RMSE.

    :param lambdas: Lista vrednosti lambda
    :param degrees: Lista stepena polinoma p
    :param rmse_results: Matrica RMSE vrednosti dimenzija (len(degrees), len(lambdas))
    :param name: Naziv .pdf fajla za cuvanje slike (akо je None, slika se ne cuva)
    """
    plt.figure(figsize=(8, 6))
    for idx, p in enumerate(degrees):
        plt.plot(lambdas, rmse_results[idx], marker='o', label=f'Степен полинома p = {p}')

    plt.xscale('log')
    plt.xlabel('Ламбда ($\lambda$)')
    plt.ylabel('Просечни RMSE (Унакрсна валидација)')
    plt.title('Избор хиперпараметара за LASSO регресију')
    plt.legend()

    _save_plot_if_needed(name)


def plot_parameter_history(w_history: np.ndarray,
                           p: int,
                           lmbda: float,
                           name: str = None,
) -> None:
    """
    Isctava konvergenciju ucenih parametara (tezina w) tokom treniranja za izabrano p i lambda.

    :param w_history: Niti/Matrica istorije parametara oblika (broj_iteracija, broj_obelezja)
    :param p: Izabrani stepen polinoma
    :param lmbda: Izabrani regularizacioni parametar lambda
    :param name: Naziv .pdf fajla za cuvanje slike (ako je None, slika se ne cuva)
    """
    plt.figure(figsize=(9, 6))
    num_iterations, num_features = w_history.shape

    for j in range(num_features):
        plt.plot(range(num_iterations), w_history[:, j], label=f'$w_{{{j + 1}}}$')

    plt.xlabel('Итерација')
    plt.ylabel('Вредност параметра ($w$)')
    plt.title(f'Промена параметрана током тренирања ($p = {p}$, $\lambda = {lmbda}$)')

    # Prikazujemo legendu samo ako nema previše obeležja radi preglednosti
    if num_features <= 15:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    _save_plot_if_needed(name)


def plot_lasso_3d_surface(X: np.ndarray,
                          y: np.ndarray,
                          w: np.ndarray,
                          b: float,
                          mean: np.ndarray,
                          std: np.ndarray,
                          best_p: int,
                          best_lambda: float,
                          name: str = "lasso_3d_prikaz.png",
) -> None:
    """
    Iscrtava 3D prikaz originalnih podataka i naučene regresione površi za LASSO model.

    :param X: Matrica originalnih obeležja (dimenzija n x 2)
    :param y: Vektor ciljne promenljive
    :param w: Težinski koeficijenti modela
    :param b: Slobodni član (bias)
    :param mean: Srednje vrednosti korišćene za skaliranje obeležja
    :param std: Standardne devijacije korišćene za skaliranje obeležja
    :param best_p: Optimalni stepen polinoma
    :param best_lambda: Optimalni regularizacioni parametar lambda
    :param name: Naziv fajla za čuvanje slike (ako je None, slika se ne čuva)
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')

    # Prikaz originalnih tačaka podataka
    ax.scatter(X[:, 0], X[:, 1], y, c=y, cmap='viridis', label='Подаци', s=30, alpha=0.8)

    # Kreiranje mreže za iscrtavanje površi
    x1_grid = np.linspace(X[:, 0].min(), X[:, 0].max(), 40)
    x2_grid = np.linspace(X[:, 1].min(), X[:, 1].max(), 40)
    XX1, XX2 = np.meshgrid(x1_grid, x2_grid)
    X_grid = np.column_stack([XX1.ravel(), XX2.ravel()])

    # Transformacija i skaliranje mrežnih tačaka korišćenjem parametara sa trening skupa
    X_grid_poly = generate_polynomial_features(X_grid, best_p)
    X_grid_scaled = (X_grid_poly - mean) / std
    ZZ = predict(X_grid_scaled, w, b).reshape(XX1.shape)

    # Iscrtavanje 3D površi
    ax.plot_surface(XX1, XX2, ZZ, cmap='plasma', alpha=0.6, edgecolor='none')

    ax.set_xlabel("Предиктор x1")
    ax.set_ylabel("Предиктор x2")
    ax.set_zlabel("Циљна променљива y")
    ax.set_title(f"LASSO Регресија (p = {best_p}, $\\lambda$ = {best_lambda})")

    # Čuvanje 3D grafika ako je naziv prosleđen
    _save_plot_if_needed(name)

    plt.show()


def plot_lasso_residuals(y: np.ndarray,
                         y_pred: np.ndarray,
                         name: str = None
) -> None:
    """
    Iscrtava analizu reziduala (razlike stvarnih i predviđenih vrednosti) za LASSO model.

    :param y: Vektor stvarnih vrednosti
    :param y_pred: Vektor predviđenih vrednosti
    :param name: Naziv fajla za čuvanje slike (ako je None, slika se ne čuva)
    """
    plt.figure(figsize=(8, 6))
    residuals = y - y_pred

    plt.scatter(y_pred, residuals, color='purple', alpha=0.7, edgecolors='k', s=40)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Нулта линија (Идеално)')
    plt.xlabel("Предвиђене вредности ($\hat{y}$)")
    plt.ylabel("Резидуали ($y - \hat{y}$)")
    plt.title("Анализа резидуала LASSO модела")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    # Korišćenje vaše pomoćne funkcije za čuvanje
    _save_plot_if_needed(name)
    plt.show()


def plot_interactive_lasso_3d(X: np.ndarray,
                              y: np.ndarray,
                              w: np.ndarray,
                              b: float,
                              mean: np.ndarray,
                              std: np.ndarray,
                              best_p: int,
                              best_lambda: float,
                              name: str = None
)-> go.Figure:
    """
    Kreira i iscrtava interaktivnu 3D LASSO regresionu površ sa originalnim podacima pomoću Plotly biblioteke.

    :param X: Matrica originalnih obeležja (dimenzija n x 2)
    :param y: Vektor ciljne promenljive
    :param w: Težinski koeficijenti modela
    :param b: Slobodni član (bias)
    :param mean: Srednje vrednosti korišćene za skaliranje obeležja
    :param std: Standardne devijacije korišćene za skaliranje obeležja
    :param best_p: Optimalni stepen polinoma
    :param best_lambda: Optimalni regularizacioni parametar lambda
    :param name: Naziv HTML fajla za čuvanje interaktivne figure (ako je None, fajl se ne čuva)
    :return: Plotly Figure objekat
    """
    # Kreiranje mreže za iscrtavanje interaktivne 3D površi
    x1_grid = np.linspace(X[:, 0].min(), X[:, 0].max(), 40)
    x2_grid = np.linspace(X[:, 1].min(), X[:, 1].max(), 40)
    XX1, XX2 = np.meshgrid(x1_grid, x2_grid)
    X_grid = np.column_stack([XX1.ravel(), XX2.ravel()])

    # Transformacija i skaliranje mrežnih tačaka korišćenjem parametara sa trening skupa
    X_grid_poly = generate_polynomial_features(X_grid, best_p)
    X_grid_scaled = (X_grid_poly - mean) / std
    ZZ = predict(X_grid_scaled, w, b).reshape(XX1.shape)

    # Definisanje 3D tačaka originalnih podataka
    scatter_trace = go.Scatter3d(
        x=X[:, 0],
        y=X[:, 1],
        z=y,
        mode='markers',
        marker=dict(
            size=5,
            color=y,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title="y")
        ),
        name='Оригинални подаци'
    )

    # Definisanje 3D površi naučenog modela
    surface_trace = go.Surface(
        x=XX1,
        y=XX2,
        z=ZZ,
        colorscale='Plasma',
        opacity=0.7,
        name='Регресиона површ'
    )

    # Kreiranje interaktivne figure
    fig_interactive = go.Figure(data=[scatter_trace, surface_trace])

    fig_interactive.update_layout(
        title=f'Интерактивна 3D LASSO регресија (p = {best_p}, $\\lambda$ = {best_lambda})',
        scene=dict(
            xaxis_title='x1',
            yaxis_title='x2',
            zaxis_title='y'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Opciono čuvanje u HTML formatu ukoliko je prosleđen naziv fajla
    if name:
        fig_interactive.write_html(name)
        print(f"[INFO] Interaktivni grafik je uspešno sačuvan kao '{name}'.")

    fig_interactive.show()
    return fig_interactive


import os
import numpy as np
import pandas as pd


def train_and_evaluate_lasso(X: np.ndarray,
                             y: np.ndarray,
                             best_p: int,
                             best_lambda: float,
                             csv_path: str = "data/lasso_rezultati_predikcije.csv",
                             history_plot_name: str = "param_history.pdf") -> dict:
    """
    Obučava konačni LASSO model, računa dodatne metrike, čuva rezultate u CSV i iscrtava istoriju parametara.

    :param X: Matrica originalnih obeležja
    :param y: Vektor ciljne promenljive
    :param best_p: Optimalni stepen polinoma
    :param best_lambda: Optimalni regularizacioni parametar lambda
    :param csv_path: Putanja za čuvanje CSV fajla sa predikcijama
    :param history_plot_name: Naziv fajla za čuvanje grafika istorije parametara
    :return: Rečnik sa obučenim parametrima, predikcijama, metrikama i skaliranjem
    """
    # 1. Obučavanje konačnog LASSO modela na celom skupu podataka
    X_poly = generate_polynomial_features(X, best_p)
    X_scaled, _, mean, std = standardize_features(X_poly)

    # Pokretanje koordinatnog spusta (podržava vraćanje 3 ili 4 vrednosti)
    w, b, w_history = coordinate_descent_lasso(
        X=X_scaled,
        y=y,
        lmbda=best_lambda,
        return_history=True,
    )

    print("NAUČENI PARAMETRI HIPOTEZE:")
    print(f"  • Slobodni član (b): {b:.6f}")
    print(f"  • Težinski koeficijenti (w): {w}\n")

    # 2. Evaluacija modela na celom skupu podataka
    y_pred = predict(X_scaled, w, b)
    final_rmse = compute_rmse(y, y_pred)
    final_mae = np.mean(np.abs(y - y_pred))

    # Izračunavanje R^2 skora
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)

    print("DODATNE EVALUACIONE METRIKE MODELA:")
    print(f"  • Konačni RMSE na celom skupu: {final_rmse:.4f}")
    print(f"  • Konačni MAE na celom skupu:  {final_mae:.4f}")
    print(f"  • R² koeficijent determinacije: {r2_score:.4f}")

    # 3. Automatsko kreiranje foldera za CSV ukoliko ne postoji
    dir_name = os.path.dirname(csv_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        print(f"[INFO] Kreiran je nedostajući folder: '{dir_name}'")

    # Čuvanje stvarnih i predviđenih vrednosti u CSV fajl
    results_df = pd.DataFrame({
        "Stvarna_vrednost_y": y,
        "Predvidjena_vrednost_y": y_pred,
        "Rezidual": y - y_pred
    })
    results_df.to_csv(csv_path, index=False)
    print(f"\n[INFO] Rezultati predikcija su uspešno sačuvani u '{csv_path}'.\n")

    # 4. Iscrtavanje istorije parametara
    plot_parameter_history(
        w_history=w_history,
        p=best_p,
        lmbda=best_lambda,
        name=history_plot_name,
    )


    return {
        "w": w,
        "b": b,
        "w_history": w_history,
        "y_pred": y_pred,
        "metrics": {"rmse": final_rmse, "mae": final_mae, "r2": r2_score},
        "mean": mean,
        "std": std
    }