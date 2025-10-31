"""
Auteur : (GROUPE 2)
"""
import numpy as np
import re
def saisir_systeme():
    """
    Permet à l'utilisateur de saisir le système d'équations
    """
    print("=== Saisie du système d'équations ===")
    print("Format: a1x1 + a2x2 + ... + anxn = b")
    print("Exemple: 2x1 + 3x2 - x3 = 5")
    print("Tapez 'Fin' pour terminer la saisie")
    print()
   
    equations = []
    i = 1
   
    while True:
        equation_str = input(f"Équation {i}: ").strip()
       
        if equation_str.lower() == 'fin':
            break
       
        if '=' not in equation_str:
            print("Erreur: Format incorrect. Utilisez '='")
            continue
           
        try:
            equations.append(equation_str)
            i += 1
        except Exception as e:
            print(f"Erreur: {e}")
   
    return equations
def extraire_variables(equations):
    """
    Extrait toutes les variables utilisées dans le système
    """
    variables = set()
   
    for eq in equations:
        # Recherche de tous les motifs xi, x1, x2, etc.
        matches = re.findall(r'x\d*', eq)
        for match in matches:
            variables.add(match)
   
    # Conversion en liste triée
    variables_list = sorted(list(variables), key=lambda x: int(x[1:]) if x[1:] else 1)
    return variables_list
def parser_equation_complet(equation_str, variables):
    """
    Parse une équation complète pour extraire tous les coefficients
    """
    n_vars = len(variables)
    coefficients = [0.0] * n_vars
    constante = 0.0
   
    # Séparation des côtés gauche et droit
    parties = equation_str.split('=')
    gauche = parties[0].strip()
    droite = parties[1].strip()
   
    # Traitement du côté droit (constante)
    try:
        constante = float(droite)
    except ValueError:
        raise ValueError(f"Constante invalide: {droite}")
   
    # Normalisation du côté gauche: ajouter des + avant les - pour faciliter le parsing
    gauche_norm = re.sub(r'(?<!\d)\s*-\s*', ' + -', gauche)
   
    # Séparation des termes
    termes = re.split(r'\s*\+\s*', gauche_norm)
   
    for terme in termes:
        terme = terme.strip()
        if not terme:
            continue
       
        # Gérer les termes négatifs
        signe = 1
        if terme.startswith('-'):
            signe = -1
            terme = terme[1:].strip()
       
        # Chercher le coefficient et la variable
        if any(var in terme for var in variables):
            # Trouver quelle variable est dans ce terme
            var_trouvee = None
            for var in variables:
                if var in terme:
                    var_trouvee = var
                    break
           
            if var_trouvee:
                # Extraire le coefficient
                coef_str = terme.replace(var_trouvee, '').strip()
               
                if not coef_str or coef_str == '+':
                    coef = 1.0
                elif coef_str == '-':
                    coef = -1.0
                else:
                    # Gérer les coefficients avec signe
                    if coef_str.startswith('+'):
                        coef_str = coef_str[1:]
                    try:
                        coef = float(coef_str)
                    except ValueError:
                        coef = 1.0
               
                coef *= signe
                idx = variables.index(var_trouvee)
                coefficients[idx] = coef
        else:
            # C'est un terme constant du côté gauche
            try:
                constante -= float(terme) * signe
            except ValueError:
                pass
   
    return coefficients, constante
def construire_matrices_complet(equations):
    """
    Construit les matrices A et b à partir des équations
    """
    # Extraire toutes les variables
    variables = extraire_variables(equations)
   
    if not variables:
        raise ValueError("Aucune variable détectée dans les équations")
   
    print(f"Variables détectées: {variables}")
   
    n_equations = len(equations)
    n_variables = len(variables)
   
    A = np.zeros((n_equations, n_variables))
    b = np.zeros(n_equations)
   
    for i, eq in enumerate(equations):
        coefficients, constante = parser_equation_complet(eq, variables)
        A[i] = coefficients
        b[i] = constante
   
    return A, b, variables
def pivot_gauss(A, b):
    """
    Résout le système Ax = b par la méthode du pivot de Gauss
    """
    print("\n" + "="*50)
    print("MÉTHODE DU PIVOT DE GAUSS")
    print("="*50)
   
    n = len(b)
    # Création de la matrice augmentée [A|b]
    Ab = np.hstack([A.copy(), b.reshape(-1, 1)])
   
    print("\nMatrice augmentée initiale [A|b]:")
    print_matrix_system(Ab)
   
    # Élimination avant
    for i in range(n):
        print(f"\n--- Étape {i+1} ---")
       
        # Recherche du pivot maximum
        max_row = i + np.argmax(np.abs(Ab[i:, i]))
        if max_row != i:
            Ab[[i, max_row]] = Ab[[max_row, i]]
            print(f"Échange L{i+1} ↔ L{max_row+1}:")
            print_matrix_system(Ab)
       
        # Vérification du pivot
        if abs(Ab[i, i]) < 1e-12:
            print("Pivot nul détecté!")
            return None
       
        # Élimination
        for j in range(i + 1, n):
            facteur = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= facteur * Ab[i, i:]
            print(f"L{j+1} ← L{j+1} - ({facteur:.3f})×L{i+1}:")
            print_matrix_system(Ab)
   
    # Vérification de la cohérence
    if abs(Ab[n-1, n-1]) < 1e-12:
        if abs(Ab[n-1, -1]) < 1e-12:
            print("Système indéterminé!")
            return None
        else:
            print("Système impossible!")
            return None
   
    # Substitution arrière
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i, i]
   
    return x
def decomposition_lu(A):
    """
    Décomposition LU de la matrice A avec pivot partiel
    """
    print("\n" + "="*50)
    print("DÉCOMPOSITION LU")
    print("="*50)
   
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy().astype(float)
    P = np.eye(n)  # Matrice de permutation
   
    print("\nMatrice initiale A:")
    print(A)
   
    for k in range(n-1):
        print(f"\n--- Étape {k+1} ---")
       
        # Recherche du pivot
        max_row = k + np.argmax(np.abs(U[k:, k]))
        if max_row != k:
            # Échange des lignes dans U
            U[[k, max_row]] = U[[max_row, k]]
            # Échange des lignes dans P
            P[[k, max_row]] = P[[max_row, k]]
            # Échange des lignes dans L (seulement la partie déjà calculée)
            if k > 0:
                L[[k, max_row], :k] = L[[max_row, k], :k]
           
            print(f"Échange L{k+1} ↔ L{max_row+1}")
       
        # Vérification du pivot
        if abs(U[k, k]) < 1e-12:
            print("Pivot nul - matrice singulière")
            return None, None, None
       
        # Élimination
        for i in range(k+1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
       
        print(f"L (après étape {k+1}):")
        print(L)
        print(f"U (après étape {k+1}):")
        print(U)
   
    return P, L, U
def resoudre_lu(P, L, U, b):
    """
    Résout le système PA = LU avec Ly = Pb et Ux = y
    """
    print("\nRésolution du système LU:")
   
    n = len(b)
   
    # Appliquer la permutation à b
    Pb = P @ b
    print(f"Pb = {Pb}")
   
    # Résolution Ly = Pb (substitution avant)
    y = np.zeros(n)
    for i in range(n):
        y[i] = Pb[i] - np.dot(L[i, :i], y[:i])
   
    print(f"Solution de Ly = Pb: y = {y}")
   
    # Résolution Ux = y (substitution arrière)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        if abs(U[i, i]) < 1e-12:
            print("Matrice singulière - pas de solution unique")
            return None
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
   
    return x
def print_matrix_system(Ab):
    """
    Affiche la matrice augmentée de manière lisible
    """
    n, m = Ab.shape
    for i in range(n):
        ligne = "["
        for j in range(m-1):
            ligne += f"{Ab[i, j]:8.3f}"
        ligne += f" | {Ab[i, -1]:8.3f}]"
        print(ligne)
def verifier_solution(A, x, b, variables):
    """
    Vérifie que la solution x satisfait Ax = b
    """
    print("\n" + "="*50)
    print("VÉRIFICATION DE LA SOLUTION")
    print("="*50)
   
    if x is None:
        print("Aucune solution à vérifier")
        return False
   
    b_calc = A @ x
    erreur = np.linalg.norm(b - b_calc)
   
    print("Solution trouvée:")
    for i, var in enumerate(variables):
        print(f"  {var} = {x[i]:.6f}")
   
    print(f"\nVérification Ax = b:")
    print(f"  b calculé = {b_calc}")
    print(f"  b donné   = {b}")
    print(f"  Erreur    = {erreur:.2e}")
   
    if erreur < 1e-10:
        print("✓ Solution validée avec succès!")
        return True
    else:
        print("✗ Solution incorrecte!")
        return False
def afficher_systeme(A, b, variables):
    """
    Affiche le système d'équations de manière lisible
    """
    print("\n" + "="*50)
    print("SYSTÈME D'ÉQUATIONS")
    print("="*50)
   
    n, m = A.shape
    for i in range(n):
        equation = ""
        for j in range(m):
            coef = A[i, j]
            if abs(coef) > 1e-10:
                if equation and coef > 0:
                    equation += " + "
                elif equation and coef < 0:
                    equation += " - "
                    coef = -coef
               
                if abs(coef) == 1:
                    equation += f"{variables[j]}"
                else:
                    equation += f"{coef:.2f}{variables[j]}"
       
        if not equation:
            equation = "0"
       
        equation += f" = {b[i]:.2f}"
        print(f"Éq {i+1}: {equation}")
def main():
    """
    Programme principal
    """
    try:
        # Saisie du système
        equations = saisir_systeme()
       
        if not equations:
            print("Aucune équation saisie!")
            return
       
        # Construction des matrices
        A, b, variables = construire_matrices_complet(equations)
       
        # Affichage du système
        afficher_systeme(A, b, variables)
       
        print(f"\nMatrice A ({A.shape[0]}x{A.shape[1]}):")
        print(A)
        print(f"\nVecteur b: {b}")
       
        # Vérification de la dimension
        if A.shape[0] != A.shape[1]:
            print("\n⚠️  Système non carré - utilisation des moindres carrés")
            try:
                x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                print(f"Solution (moindres carrés):")
                for i, var in enumerate(variables):
                    print(f"  {var} = {x[i]:.6f}")
                if residuals:
                    print(f"Résidu: {residuals[0]:.2e}")
            except Exception as e:
                print(f"Erreur: {e}")
            return
       
        # Choix de la méthode
        print("\n" + "="*50)
        print("CHOIX DE LA MÉTHODE")
        print("="*50)
        print("1. Pivot de Gauss")
        print("2. Décomposition LU")
       
        choix = input("\nVotre choix (1 ou 2): ").strip()
       
        if choix == "1":
            # Méthode du pivot de Gauss
            x = pivot_gauss(A, b)
            verifier_solution(A, x, b, variables)
           
        elif choix == "2":
            # Méthode LU
            P, L, U = decomposition_lu(A)
           
            if P is not None:
                print(f"\nMatrice de permutation P:")
                print(P)
                print(f"\nMatrice triangulaire inférieure L:")
                print(L)
                print(f"\nMatrice triangulaire supérieure U:")
                print(U)
               
                x = resoudre_lu(P, L, U, b)
                verifier_solution(A, x, b, variables)
            else:
                print("La décomposition LU a échoué!")
               
        else:
            print("Choix invalide! Utilisation du pivot de Gauss par défaut.")
            x = pivot_gauss(A, b)
            verifier_solution(A, x, b, variables)
           
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        print("Veuillez vérifier le format de vos équations.")
if __name__ == "__main__":
    main()