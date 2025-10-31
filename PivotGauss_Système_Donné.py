"""
Auteur : (GROUPE 2)
"""
import numpy as np
def pivot_gauss_systeme_specifique():
    """
    Résout le système spécifique par la méthode du pivot de Gauss:
    x + y = 1
    x - y = 1
    3x + y = 3
    """
    print("=" * 60)
    print("RÉSOLUTION DU SYSTÈME PAR PIVOT DE GAUSS")
    print("=" * 60)
    print("Système à résoudre:")
    print("x + y = 1")
    print("x - y = 1")
    print("3x + y = 3")
    print()
   
    # Matrice des coefficients A (3 équations, 2 inconnues)
    A = np.array([
        [1, 1],    # Équation 1: x + y
        [1, -1],   # Équation 2: x - y
        [3, 1]     # Équation 3: 3x + y
    ], dtype=float)
   
    # Vecteur des constantes b
    b = np.array([1, 1, 3], dtype=float)
   
    print("Matrice des coefficients A:")
    print(A)
    print("\nVecteur des constantes b:")
    print(b)
   
    # Création de la matrice augmentée [A|b]
    Ab = np.column_stack([A, b])
    print("\n" + "=" * 40)
    print("MATRICE AUGMENTÉE INITIALE [A|b]:")
    print("=" * 40)
    print_matrix_augmentee(Ab)
   
    # Étape 1: Utiliser la première équation comme pivot
    print("\n" + "=" * 40)
    print("ÉTAPE 1 - PIVOT SUR LA PREMIÈRE LIGNE")
    print("=" * 40)
   
    pivot_ligne = 0  # Première ligne comme pivot
    pivot_valeur = Ab[pivot_ligne, 0]
   
    print(f"Pivot: L1, valeur = {pivot_valeur}")
   
    # Éliminer x de la deuxième équation
    ligne_cible = 1
    facteur = Ab[ligne_cible, 0] / pivot_valeur
    print(f"\nÉlimination de x dans L2:")
    print(f"Facteur = {Ab[ligne_cible, 0]} / {pivot_valeur} = {facteur}")
   
    Ab[ligne_cible] = Ab[ligne_cible] - facteur * Ab[pivot_ligne]
    print("Nouvelle L2 = L2 - (facteur) × L1")
    print_matrix_augmentee(Ab)
   
    # Éliminer x de la troisième équation
    ligne_cible = 2
    facteur = Ab[ligne_cible, 0] / pivot_valeur
    print(f"\nÉlimination de x dans L3:")
    print(f"Facteur = {Ab[ligne_cible, 0]} / {pivot_valeur} = {facteur}")
   
    Ab[ligne_cible] = Ab[ligne_cible] - facteur * Ab[pivot_ligne]
    print("Nouvelle L3 = L3 - (facteur) × L1")
    print_matrix_augmentee(Ab)
   
    # Étape 2: Utiliser la deuxième équation comme pivot pour y
    print("\n" + "=" * 40)
    print("ÉTAPE 2 - PIVOT SUR LA DEUXIÈME LIGNE")
    print("=" * 40)
   
    pivot_ligne = 1  # Deuxième ligne comme pivot
    pivot_valeur = Ab[pivot_ligne, 1]
   
    print(f"Pivot: L2, valeur = {pivot_valeur}")
   
    # Éliminer y de la troisième équation
    ligne_cible = 2
    facteur = Ab[ligne_cible, 1] / pivot_valeur
    print(f"\nÉlimination de y dans L3:")
    print(f"Facteur = {Ab[ligne_cible, 1]} / {pivot_valeur} = {facteur}")
   
    Ab[ligne_cible] = Ab[ligne_cible] - facteur * Ab[pivot_ligne]
    print("Nouvelle L3 = L3 - (facteur) × L2")
    print_matrix_augmentee(Ab)
   
    # Vérification du système
    print("\n" + "=" * 40)
    print("VÉRIFICATION DU SYSTÈME")
    print("=" * 40)
   
    # La troisième équation devrait donner 0 = 0
    eq3_constante = Ab[2, 2]
    print(f"Troisième équation après élimination: 0 = {eq3_constante}")
   
    if abs(eq3_constante) < 1e-10:
        print("✓ La troisième équation est cohérente (0 = 0)")
        print("✓ Le système est compatible")
    else:
        print("✗ La troisième équation est incohérente")
        print("✗ Le système n'a pas de solution")
        return None
   
    # Résolution par substitution arrière
    print("\n" + "=" * 40)
    print("SUBSTITUTION ARRIÈRE")
    print("=" * 40)
   
    # À ce stade, nous avons:
    # L1: x + y = 1
    # L2: -2y = 0
   
    # Résolution pour y à partir de L2
    y = Ab[1, 2] / Ab[1, 1]
    print(f"Calcul de y:")
    print(f"y = {Ab[1, 2]} / {Ab[1, 1]} = {y}")
   
    # Résolution pour x à partir de L1
    x = (Ab[0, 2] - Ab[0, 1] * y) / Ab[0, 0]
    print(f"\nCalcul de x:")
    print(f"x = ({Ab[0, 2]} - {Ab[0, 1]} × {y}) / {Ab[0, 0]} = {x}")
   
    return x, y
def print_matrix_augmentee(Ab):
    """
    Affiche la matrice augmentée de manière formatée
    """
    n, m = Ab.shape
    print("[" + " "*3, end="")
    for j in range(m-1):
        print(f"x{j+1:8}", end="")
    print(f" |{'constante':>10}]")
   
    for i in range(n):
        ligne = f"L{i+1} ["
        for j in range(m-1):
            ligne += f"{Ab[i, j]:8.2f}"
        ligne += f" | {Ab[i, -1]:10.2f}]"
        print(ligne)
def verifier_solution(x, y):
    """
    Vérifie que la solution satisfait toutes les équations
    """
    print("\n" + "=" * 50)
    print("VÉRIFICATION DE LA SOLUTION")
    print("=" * 50)
   
    print(f"Solution trouvée: x = {x:.6f}, y = {y:.6f}")
    print()
   
    # Vérification de chaque équation
    eq1 = x + y
    eq2 = x - y
    eq3 = 3*x + y
   
    print("Équation 1: x + y = 1")
    print(f"  {x:.6f} + {y:.6f} = {eq1:.6f} → {'✓' if abs(eq1 - 1) < 1e-10 else '✗'}")
   
    print("Équation 2: x - y = 1")
    print(f"  {x:.6f} - {y:.6f} = {eq2:.6f} → {'✓' if abs(eq2 - 1) < 1e-10 else '✗'}")
   
    print("Équation 3: 3x + y = 3")
    print(f"  3×{x:.6f} + {y:.6f} = {eq3:.6f} → {'✓' if abs(eq3 - 3) < 1e-10 else '✗'}")
   
    # Vérification globale
    erreur_totale = abs(eq1 - 1) + abs(eq2 - 1) + abs(eq3 - 3)
    print(f"\nErreur totale: {erreur_totale:.2e}")
   
    if erreur_totale < 1e-10:
        print("\n🎉 SOLUTION VALIDÉE AVEC SUCCÈS!")
        print(f"✅ x = {x}")
        print(f"✅ y = {y}")
    else:
        print("\n❌ Solution incorrecte!")
def main():
    """
    Programme principal
    """
    print("SYSTÈME D'ÉQUATIONS LINÉAIRES")
    print("Méthode: Pivot de Gauss")
    print()
   
    # Résolution du système
    resultat = pivot_gauss_systeme_specifique()
   
    if resultat is not None:
        x, y = resultat
        verifier_solution(x, y)
       
        # Affichage final
        print("\n" + "=" * 60)
        print("RÉSULTAT FINAL")
        print("=" * 60)
        print(f"La solution du système est:")
        print(f"  x = {x}")
        print(f"  y = {y}")
       
        # Vérification numérique précise
        print("\nVérification numérique précise:")
        print(f"  x + y  = {x + y} (devrait être 1)")
        print(f"  x - y  = {x - y} (devrait être 1)")
        print(f"  3x + y = {3*x + y} (devrait être 3)")
    else:
        print("Le système n'a pas de solution unique.")
if __name__ == "__main__":
    main()