"""
Auteur : (GROUPE 2)
"""
# ------------------------------------------------------------
# Programme : Méthodes numériques pour l'approximation de √N.
# Objectif  : Implémentation de 4 méthodes numériques.
# ------------------------------------------------------------
def f(x, N):
    """Fonction dont on cherche la racine : f(x) = x² - N"""
    return x**2 - N
def df(x, N):
    """Dérivée de la fonction f(x) = x² - N : f'(x) = 2x"""
    return 2 * x
# =============================================================================
# MÉTHODE DE DICHOTOMIE
# =============================================================================
def dichotomie(N, a, b, epsilon=0.1, afficher_tableau=True):
    """
    Approximation de √N par la méthode de la dichotomie
   
    Paramètres:
    N: nombre dont on cherche la racine
    a, b: bornes de l'intervalle initial
    epsilon: précision souhaitée
    afficher_tableau: booléen pour afficher le tableau de suivi
    """
    # Vérification de la validité de l'intervalle
    if f(a, N) * f(b, N) > 0:
        print("❌ La méthode n'est pas applicable : f(a) et f(b) ont le même signe.")
        return None
   
    if afficher_tableau:
        print("\n*-----------------------------------------------------------------------------*")
        print("| Étape |      a      |      b      |      m      |     f(m)     | Intervalle |")
        print("*-----------------------------------------------------------------------------*")
   
    etape = 0
    while (b - a) > epsilon:
        # Calcul du point milieu
        m = (a + b) / 2
        fm = f(m, N)
       
        if afficher_tableau:
            print(f"| {etape:^5d} | {a:^10.4f} | {b:^10.4f} | {m:^10.4f} | {fm:^11.4f} | [{a:.2f}, {b:.2f}] |")
       
        # Vérification du signe pour savoir dans quelle moitié chercher
        if f(a, N) * fm < 0:
            b = m   # La racine est dans [a, m]
        else:
            a = m   # La racine est dans [m, b]
        etape += 1
   
    racine_approx = (a + b) / 2
   
    if afficher_tableau:
        print("*---------------------------------------------------------------------------*")
        print(f"\nLa racine carrée de {N} est approximativement : {round(racine_approx, 1)}")
        print(f"Intervalle final : [{a:.4f}, {b:.4f}]")
        print(f"Valeur numérique obtenue par la méthode Dichotomique vaut : {racine_approx:.6f}")
   
    return racine_approx
# =============================================================================
# MÉTHODE DE BALAYAGE
# =============================================================================
def balayage(N, a, b, pas=0.1, afficher_tableau=True):
    """
    Approximation de √N par la méthode du balayage
   
    Paramètres:
    N: nombre dont on cherche la racine
    a, b: bornes de l'intervalle initial
    pas: pas de balayage (correspond à la précision)
    afficher_tableau: booléen pour afficher le tableau de suivi
    """
    # Vérification de la validité de l'intervalle
    if f(a, N) * f(b, N) > 0:
        print("❌ La méthode n'est pas applicable : f(a) et f(b) ont le même signe.")
        return None
   
    if afficher_tableau:
        print("\n-----------------------------------------------------------------")
        print("|   x    |    f(x)     |   f(x+pas)  |  Changement de signe ?   |")
        print("-----------------------------------------------------------------")
   
    x = a
    racine_approx = None
   
    while x + pas <= b:
        fx = f(x, N)
        fx_next = f(x + pas, N)
       
        # Vérifie s'il y a un changement de signe
        signe_change = fx * fx_next < 0
       
        if afficher_tableau:
            print(f"| {x:^7.2f} | {fx:^11.4f} | {fx_next:^11.4f} | {'Oui' if signe_change else 'Non':^23} |")
       
        # Si changement de signe → la racine est entre x et x+pas
        if signe_change:
            racine_approx = (x + (x + pas)) / 2
            break
       
        x += pas
   
    if afficher_tableau:
        print("-----------------------------------------------------------------")
        if racine_approx is not None:
            print(f"\nLa racine de {N} se trouve dans l'intervalle : [{x:.1f}, {x + pas:.1f}]")
            print(f"Valeur approchée de √{N} à 10^-1 près par la méthode de Balayage vaut : {round(racine_approx, 1)}")
        else:
            print("Aucune racine trouvée dans l'intervalle spécifié.")
   
    return racine_approx
# =============================================================================
# MÉTHODE DE NEWTON-RAPHSON
# =============================================================================
def newton(N, x0, epsilon=1e-6, max_iterations=100, afficher_tableau=True):
    """
    Approximation de √N par la méthode de Newton-Raphson
   
    Paramètres:
    N: nombre dont on cherche la racine
    x0: approximation initiale
    epsilon: critère d'arrêt sur la différence entre itérations
    max_iterations: nombre maximum d'itérations
    afficher_tableau: booléen pour afficher le tableau de suivi
    """
    if afficher_tableau:
        print("\n*--------------------------------------------------------------*")
        print("| Étape |      x_n      |     f(x_n)    |     Erreur     |")
        print("*--------------------------------------------------------------*")
   
    x_prev = x0
    for i in range(max_iterations):
        # Formule de Newton : x_{n+1} = x_n - f(x_n)/f'(x_n)
        x_next = x_prev - f(x_prev, N) / df(x_prev, N)
       
        erreur = abs(x_next - x_prev)
       
        if afficher_tableau:
            print(f"| {i:^5d} | {x_prev:^12.6f} | {f(x_prev, N):^12.6f} | {erreur:^13.6e} |")
       
        # Critère d'arrêt
        if erreur < epsilon:
            break
       
        x_prev = x_next
   
    if afficher_tableau:
        print("*--------------------------------------------------------------*")
        print(f"\nValeur approchée de √{N} par la méthode de Newton : {x_next:.6f}")
        print(f"Nombre d'itérations : {i+1}")
   
    return x_next
# =============================================================================
# MÉTHODE DE LA SÉCANTE
# =============================================================================
def secante(N, x0, x1, epsilon=1e-6, max_iterations=100, afficher_tableau=True):
    """
    Approximation de √N par la méthode de la sécante
   
    Paramètres:
    N: nombre dont on cherche la racine
    x0, x1: deux approximations initiales
    epsilon: critère d'arrêt sur la différence entre itérations
    max_iterations: nombre maximum d'itérations
    afficher_tableau: booléen pour afficher le tableau de suivi
    """
    if afficher_tableau:
        print("\n*-------------------------------------------------------------------*")
        print("| Étape |      x_n      |     f(x_n)    |     Erreur     |")
        print("*-------------------------------------------------------------------*")
   
    x_prev2 = x0  # x_{n-1}
    x_prev1 = x1  # x_n
   
    # Première itération
    f_prev2 = f(x_prev2, N)
   
    for i in range(max_iterations):
        f_prev1 = f(x_prev1, N)
       
        # Formule de la sécante : x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))
        if abs(f_prev1 - f_prev2) < 1e-12:  # Éviter la division par zéro
            break
           
        x_next = x_prev1 - f_prev1 * (x_prev1 - x_prev2) / (f_prev1 - f_prev2)
       
        erreur = abs(x_next - x_prev1)
       
        if afficher_tableau:
            print(f"| {i:^5d} | {x_prev1:^12.6f} | {f_prev1:^12.6f} | {erreur:^13.6e} |")
       
        # Critère d'arrêt
        if erreur < epsilon:
            break
       
        # Mise à jour pour l'itération suivante
        x_prev2, x_prev1 = x_prev1, x_next
        f_prev2 = f_prev1
   
    if afficher_tableau:
        print("*-------------------------------------------------------------------*")
        print(f"\nValeur approchée de √{N} par la méthode de la Sécante : {x_next:.6f}")
        print(f"Nombre d'itérations : {i+1}")
   
    return x_next
# =============================================================================
# FONCTION PRINCIPALE POUR TESTER TOUTES LES MÉTHODES
# =============================================================================
def comparer_methodes():
    """Fonction pour tester et comparer les 4 méthodes"""
    print("=" * 70)
    print("COMPARAISON DES MÉTHODES NUMÉRIQUES POUR LE CALCUL DE √N")
    print("=" * 70)
   
    # Lecture des données
    N = float(input("Entrez le nombre dont vous voulez la racine carrée : "))
   
    print(f"\nCalcul de √{N} :")
    print(f"Valeur exacte (approchée) : {N**0.5:.6f}")
    print("-" * 50)
   
    # Test de la dichotomie
    print("\n1. MÉTHODE DE DICHOTOMIE")
    a = float(input("Borne inférieure a : "))
    b = float(input("Borne supérieure b : "))
    result_dicho = dichotomie(N, a, b)
   
    # Test du balayage
    print("\n2. MÉTHODE DE BALAYAGE")
    result_balayage = balayage(N, a, b)
   
    # Test de Newton
    print("\n3. MÉTHODE DE NEWTON-RAPHSON")
    x0_newton = float(input("Valeur initiale x0 pour Newton : "))
    result_newton = newton(N, x0_newton, epsilon=1e-6)
   
    # Test de la sécante
    print("\n4. MÉTHODE DE LA SÉCANTE")
    x0_secante = float(input("Première valeur initiale x0 pour la sécante : "))
    x1_secante = float(input("Deuxième valeur initiale x1 pour la sécante : "))
    result_secante = secante(N, x0_secante, x1_secante, epsilon=1e-6)
   
    # Affichage comparatif
    print("\n" + "=" * 70)
    print("RÉCAPITULATIF DES RÉSULTATS")
    print("=" * 70)
    print(f"Valeur de référence : {N**0.5:.6f}")
    print(f"Dichotomie  : {result_dicho:.6f}" if result_dicho else "Dichotomie  : Non applicable")
    print(f"Balayage    : {result_balayage:.6f}" if result_balayage else "Balayage    : Non applicable")
    print(f"Newton      : {result_newton:.6f}")
    print(f"Sécante     : {result_secante:.6f}")
# =============================================================================
# EXÉCUTION
# =============================================================================
if __name__ == "__main__":
    # Vous pouvez choisir d'exécuter une méthode spécifique ou la comparaison
    print("Choisissez une option :")
    print("1 - Méthode de Dichotomie")
    print("2 - Méthode de Balayage")
    print("3 - Méthode de Newton")
    print("4 - Méthode de la Sécante")
    print("5 - Comparer toutes les méthodes")
   
    choix = input("Entrez Votre choix (1-5) : ")
   
    if choix == "1":
        N = float(input("Entrez le nombre N dont vous voulez la racine carrée : "))
        a = float(input("Entrez la Borne inférieure a : "))
        b = float(input("Entrez la Borne supérieure b : "))
        dichotomie(N, a, b)
    elif choix == "2":
        N = float(input("Entrez N dont vous voulez la racine carrée: "))
        a = float(input("Entrez la Borne inférieure a : "))
        b = float(input("Entrez la Borne supérieure b : "))
        balayage(N, a, b)
    elif choix == "3":
        N = float(input("Entrez N dont vous voulez la racine carrée: "))
        x0 = float(input("Entrez laValeur initiale x0 : "))
        newton(N, x0)
    elif choix == "4":
        N = float(input("Entrez N dont vous voulez la racine carrée: "))
        x0 = float(input("Entrez la Première valeur initiale x0 : "))
        x1 = float(input("Entrez la Deuxième valeur initiale x1 : "))
        secante(N, x0, x1)
    elif choix == "5":
        comparer_methodes()
    else:
        print("Choix invalide!")