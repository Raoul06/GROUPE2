"""
Programme : Méthodes Numériques avec SciPy/NumPy - Version Complète
Méthodes implémentées : Dichotomie, Balayage, Newton, Sécante avec fonctions natives
Auteur : (GROUPE 2)
"""
import numpy as np
from scipy.optimize import bisect, newton, root_scalar
import matplotlib.pyplot as plt
# =============================================================================
# CONFIGURATION DES GRAPHIQUES POUR SPYDER
# =============================================================================
def configurer_graphiques():
    """Configure l'affichage des graphiques pour Spyder"""
    try:
        # Configuration pour Spyder
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        print("✅ Configuration graphique terminée")
    except Exception as e:
        print(f"⚠️  Attention configuration graphique : {e}")
# Appeler la configuration au début
configurer_graphiques()
# =============================================================================
# FONCTIONS MATHÉMATIQUES
# =============================================================================
def f(x, N):
    """Fonction dont on cherche la racine : f(x) = x² - N"""
    return x**2 - N
def df(x, N):
    """Dérivée de la fonction : f'(x) = 2x"""
    return 2 * x
# =============================================================================
# MÉTHODE DE DICHOTOMIE AVEC SCIPY
# =============================================================================
def dichotomie_scipy():
    """
    Méthode de dichotomie utilisant scipy.optimize.bisect
    Cette méthode est robuste et garantie de converger
    """
    print("\n" + "="*60)
    print("DICHOTOMIE AVEC SCIPY.OPTIMIZE.BISECT")
    print("="*60)
   
    # Lecture des paramètres
    N = float(input("Entrez le nombre N dont vous voulez la racine carrée : "))
    a = float(input("Entrez la borne inférieure a : "))
    b = float(input("Entrez la borne supérieure b : "))
   
    print(f"\n🔍 Recherche de √{N} sur l'intervalle [{a}, {b}]")
   
    try:
        # Utilisation de la fonction bisect de SciPy
        racine = bisect(lambda x: f(x, N), a, b)
        valeur_exacte = np.sqrt(N)
       
        print(f"\n✅ RÉSULTATS DICHOTOMIE :")
        print(f"• Racine approximative : {racine:.10f}")
        print(f"• Valeur exacte NumPy  : {valeur_exacte:.10f}")
        print(f"• Erreur absolue       : {abs(racine - valeur_exacte):.2e}")
        print(f"• Intervalle final     : [{a}, {b}]")
       
        # Visualisation
        visualiser_dichotomie(N, a, b, racine)
       
        return racine
       
    except ValueError as e:
        print(f"❌ Erreur : {e}")
        print("💡 Vérifiez que f(a) et f(b) ont des signes opposés")
        return None
def visualiser_dichotomie(N, a, b, racine):
    """Visualise le processus de dichotomie"""
    x = np.linspace(a, b, 500)
    y = f(x, N)
   
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2, label=f'f(x) = x² - {N}')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.axvline(x=racine, color='red', linestyle='--', alpha=0.7, label=f'Racine ≈ {racine:.6f}')
    plt.plot(racine, 0, 'ro', markersize=8, label='Solution')
   
    # Marquer l'intervalle initial
    plt.axvline(x=a, color='green', linestyle=':', alpha=0.6, label=f'Borne a = {a}')
    plt.axvline(x=b, color='green', linestyle=':', alpha=0.6, label=f'Borne b = {b}')
   
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Méthode de Dichotomie pour √{N}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show(block=True)
# =============================================================================
# MÉTHODE DE BALAYAGE AVEC NUMPY
# =============================================================================
def balayage_numpy():
    """
    Méthode de balayage utilisant NumPy
    Parcourt l'intervalle avec un pas fixe pour détecter les changements de signe
    """
    print("\n" + "="*60)
    print("BALAYAGE AVEC NUMPY")
    print("="*60)
   
    # Lecture des paramètres
    N = float(input("Entrez le nombre N dont vous voulez la racine carrée : "))
    a = float(input("Entrez la borne inférieure a : "))
    b = float(input("Entrez la borne supérieure b : "))
    n_points = int(input("Nombre de points pour le balayage (défaut 1000) : ") or "1000")
   
    print(f"\n🔍 Balayage de √{N} sur [{a}, {b}] avec {n_points} points")
   
    try:
        # Création d'un array de points régulièrement espacés
        x_values = np.linspace(a, b, n_points)
        f_values = f(x_values, N)
       
        # Recherche des changements de signe
        sign_changes = np.where(f_values[:-1] * f_values[1:] < 0)[0]
       
        if len(sign_changes) > 0:
            # Premier changement de signe trouvé
            idx = sign_changes[0]
            x_left = x_values[idx]
            x_right = x_values[idx + 1]
            racine_approx = (x_left + x_right) / 2
            valeur_exacte = np.sqrt(N)
           
            print(f"\n✅ RÉSULTATS BALAYAGE :")
            print(f"• Racine approximative : {racine_approx:.10f}")
            print(f"• Valeur exacte NumPy  : {valeur_exacte:.10f}")
            print(f"• Erreur absolue       : {abs(racine_approx - valeur_exacte):.2e}")
            print(f"• Intervalle détecté   : [{x_left:.6f}, {x_right:.6f}]")
            print(f"• Largeur intervalle   : {x_right - x_left:.6f}")
           
            # Visualisation
            visualiser_balayage(N, a, b, x_values, f_values, sign_changes, racine_approx)
           
            return racine_approx
        else:
            print("❌ Aucun changement de signe détecté dans l'intervalle")
            return None
           
    except Exception as e:
        print(f"❌ Erreur : {e}")
        return None
def visualiser_balayage(N, a, b, x_values, f_values, sign_changes, racine_approx):
    """Visualise le processus de balayage"""
    plt.figure(figsize=(12, 6))
   
    # Graphique principal
    plt.subplot(1, 2, 1)
    plt.plot(x_values, f_values, 'b-', linewidth=1.5, label=f'f(x) = x² - {N}')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.axvline(x=racine_approx, color='red', linestyle='--', alpha=0.7)
    plt.plot(racine_approx, 0, 'ro', markersize=6, label=f'Racine ≈ {racine_approx:.6f}')
   
    # Marquer les changements de signe
    for idx in sign_changes[:3]:  # Maximum 3 premiers changements
        x_left = x_values[idx]
        x_right = x_values[idx + 1]
        plt.axvspan(x_left, x_right, alpha=0.2, color='orange', label='Changement de signe' if idx == sign_changes[0] else "")
   
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Méthode de Balayage pour √{N}')
    plt.legend()
    plt.grid(True, alpha=0.3)
   
    # Zoom sur la racine
    plt.subplot(1, 2, 2)
    if len(sign_changes) > 0:
        idx = sign_changes[0]
        x_left = x_values[idx]
        x_right = x_values[idx + 1]
        marge = (x_right - x_left) * 0.5
        x_zoom = np.linspace(x_left - marge, x_right + marge, 200)
        y_zoom = f(x_zoom, N)
       
        plt.plot(x_zoom, y_zoom, 'g-', linewidth=2, label=f'f(x) = x² - {N} (zoom)')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.axvline(x=racine_approx, color='red', linestyle='--', alpha=0.7)
        plt.plot(racine_approx, 0, 'ro', markersize=6, label=f'Racine ≈ {racine_approx:.6f}')
        plt.axvline(x=x_left, color='orange', linestyle=':', alpha=0.6, label='Intervalle détection')
        plt.axvline(x=x_right, color='orange', linestyle=':', alpha=0.6)
        plt.fill_betweenx(y_zoom, x_left, x_right, alpha=0.2, color='orange')
       
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Zoom sur la racine')
        plt.legend()
        plt.grid(True, alpha=0.3)
   
    plt.tight_layout()
    plt.show(block=True)
# =============================================================================
# MÉTHODE DE NEWTON-RAPHSON AVEC SCIPY
# =============================================================================
def newton_scipy():
    """
    Méthode de Newton-Raphson utilisant scipy.optimize.newton
    Convergence quadratique quand elle fonctionne
    """
    print("\n" + "="*60)
    print("NEWTON-RAPHSON AVEC SCIPY.OPTIMIZE.NEWTON")
    print("="*60)
   
    # Lecture des paramètres
    N = float(input("Entrez le nombre N dont vous voulez la racine carrée : "))
    x0 = float(input("Entrez la valeur initiale x0 : "))
   
    print(f"\n🔍 Newton-Raphson pour √{N} avec x0 = {x0}")
   
    try:
        # Newton avec dérivée fournie (convergence plus rapide)
        racine = newton(lambda x: f(x, N), x0, fprime=lambda x: df(x, N))
        valeur_exacte = np.sqrt(N)
       
        print(f"\n✅ RÉSULTATS NEWTON-RAPHSON :")
        print(f"• Racine approximative : {racine:.10f}")
        print(f"• Valeur exacte NumPy  : {valeur_exacte:.10f}")
        print(f"• Erreur absolue       : {abs(racine - valeur_exacte):.2e}")
        print(f"• Erreur relative      : {abs((racine - valeur_exacte)/valeur_exacte):.2e}")
       
        # Visualisation
        visualiser_newton(N, x0, racine)
       
        return racine
       
    except Exception as e:
        print(f"❌ Erreur : {e}")
        print("💡 Essayez une autre valeur initiale x0")
        return None
def visualiser_newton(N, x0, racine):
    """Visualise le processus de Newton-Raphson"""
    a, b = max(0, racine - 1), racine + 1  # Intervalle autour de la racine
    x = np.linspace(a, b, 500)
    y = f(x, N)
   
    plt.figure(figsize=(10, 6))
   
    # Fonction et racine
    plt.plot(x, y, 'b-', linewidth=2, label=f'f(x) = x² - {N}')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.axvline(x=racine, color='red', linestyle='--', alpha=0.7, label=f'Racine ≈ {racine:.6f}')
    plt.plot(racine, 0, 'ro', markersize=8, label='Solution')
   
    # Point initial
    plt.plot(x0, f(x0, N), 'go', markersize=8, label=f'Point initial x0 = {x0}')
    plt.axvline(x=x0, color='green', linestyle=':', alpha=0.6)
   
    # Tangente au point initial (illustration)
    x_tangent = np.linspace(x0 - 0.5, x0 + 0.5, 100)
    tangent = f(x0, N) + df(x0, N) * (x_tangent - x0)
    plt.plot(x_tangent, tangent, 'g--', alpha=0.7, label='Tangente en x0')
   
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Méthode de Newton-Raphson pour √{N}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show(block=True)
# =============================================================================
# MÉTHODE DE LA SÉCANTE AVEC SCIPY
# =============================================================================
def secante_scipy():
    """
    Méthode de la sécante utilisant scipy.optimize.newton sans dérivée
    Alternative à Newton quand la dérivée n'est pas disponible
    """
    print("\n" + "="*60)
    print("MÉTHODE DE LA SÉCANTE AVEC SCIPY")
    print("="*60)
   
    # Lecture des paramètres
    N = float(input("Entrez le nombre N dont vous voulez la racine carrée : "))
    x0 = float(input("Entrez la première valeur initiale x0 : "))
    x1 = float(input("Entrez la deuxième valeur initiale x1 : "))
   
    print(f"\n🔍 Sécante pour √{N} avec x0 = {x0}, x1 = {x1}")
   
    try:
        # Newton sans dérivée = méthode de la sécante
        racine = newton(lambda x: f(x, N), x0)
        valeur_exacte = np.sqrt(N)
       
        print(f"\n✅ RÉSULTATS SÉCANTE :")
        print(f"• Racine approximative : {racine:.10f}")
        print(f"• Valeur exacte NumPy  : {valeur_exacte:.10f}")
        print(f"• Erreur absolue       : {abs(racine - valeur_exacte):.2e}")
        print(f"• Erreur relative      : {abs((racine - valeur_exacte)/valeur_exacte):.2e}")
       
        # Visualisation
        visualiser_secante(N, x0, x1, racine)
       
        return racine
       
    except Exception as e:
        print(f"❌ Erreur : {e}")
        print("💡 Essayez d'autres valeurs initiales x0 et x1")
        return None
def visualiser_secante(N, x0, x1, racine):
    """Visualise le processus de la sécante"""
    a, b = min(x0, x1, racine) - 0.5, max(x0, x1, racine) + 0.5
    x = np.linspace(a, b, 500)
    y = f(x, N)
   
    plt.figure(figsize=(10, 6))
   
    # Fonction et racine
    plt.plot(x, y, 'b-', linewidth=2, label=f'f(x) = x² - {N}')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.axvline(x=racine, color='red', linestyle='--', alpha=0.7, label=f'Racine ≈ {racine:.6f}')
    plt.plot(racine, 0, 'ro', markersize=8, label='Solution')
   
    # Points initiaux
    plt.plot(x0, f(x0, N), 'go', markersize=8, label=f'x0 = {x0}')
    plt.plot(x1, f(x1, N), 'mo', markersize=8, label=f'x1 = {x1}')
   
    # Sécante entre x0 et x1 (illustration)
    x_secante = np.linspace(min(x0, x1), max(x0, x1), 100)
    pente = (f(x1, N) - f(x0, N)) / (x1 - x0) if x1 != x0 else 0
    secante_line = f(x0, N) + pente * (x_secante - x0)
    plt.plot(x_secante, secante_line, 'm--', alpha=0.7, label='Sécante')
   
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Méthode de la Sécante pour √{N}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show(block=True)
# =============================================================================
# COMPARAISON VISUELLE DE TOUTES LES MÉTHODES
# =============================================================================
def comparer_methodes_visuelle():
    """
    Compare visuellement les 4 méthodes sur un même graphique
    Permet de voir les différences de convergence et de précision
    """
    print("\n" + "="*60)
    print("COMPARAISON VISUELLE DES 4 MÉTHODES")
    print("="*60)
   
    # Paramètres communs
    N = float(input("Entrez le nombre N : "))
    a = max(0.1, float(input("Borne inférieure a : ")))
    b = float(input("Borne supérieure b : "))
    x0 = float(input("Valeur initiale pour Newton/Sécante : "))
   
    valeur_exacte = np.sqrt(N)
    print(f"\n🔍 Comparaison pour √{N} = {valeur_exacte:.6f}")
   
    # Calcul avec les 4 méthodes
    resultats = {}
   
    try:
        resultats['Dichotomie'] = bisect(lambda x: f(x, N), a, b)
        print("✅ Dichotomie terminée")
    except: pass
   
    try:
        # Balayage simplifié
        x_vals = np.linspace(a, b, 1000)
        f_vals = f(x_vals, N)
        sign_changes = np.where(f_vals[:-1] * f_vals[1:] < 0)[0]
        if len(sign_changes) > 0:
            idx = sign_changes[0]
            resultats['Balayage'] = (x_vals[idx] + x_vals[idx + 1]) / 2
        print("✅ Balayage terminé")
    except: pass
   
    try:
        resultats['Newton'] = newton(lambda x: f(x, N), x0, fprime=lambda x: df(x, N))
        print("✅ Newton terminé")
    except: pass
   
    try:
        resultats['Sécante'] = newton(lambda x: f(x, N), x0)
        print("✅ Sécante terminée")
    except: pass
   
    # Visualisation comparative
    visualiser_comparaison(N, a, b, resultats, valeur_exacte)
   
    # Affichage des résultats
    print(f"\n📊 RÉSULTATS COMPARATIFS :")
    print("-" * 65)
    print(f"{'Méthode':<12} | {'Valeur':<15} | {'Erreur absolue':<15}")
    print("-" * 65)
   
    for methode, valeur in resultats.items():
        erreur = abs(valeur - valeur_exacte)
        print(f"{methode:<12} | {valeur:<15.10f} | {erreur:<15.2e}")
   
    print(f"{'EXACT':<12} | {valeur_exacte:<15.10f} | {'0':<15}")
    print("-" * 65)
   
    return resultats
def visualiser_comparaison(N, a, b, resultats, valeur_exacte):
    """Visualise la comparaison des 4 méthodes"""
    x = np.linspace(a, b, 500)
    y = f(x, N)
   
    plt.figure(figsize=(14, 8))
   
    # Fonction principale
    plt.plot(x, y, 'b-', linewidth=2, label=f'f(x) = x² - {N}')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='y = 0')
   
    # Couleurs pour chaque méthode
    colors = {'Dichotomie': 'red', 'Balayage': 'green', 'Newton': 'orange', 'Sécante': 'purple'}
   
    # Marquer les racines de chaque méthode
    for methode, valeur in resultats.items():
        couleur = colors.get(methode, 'blue')
        plt.axvline(x=valeur, color=couleur, linestyle='--', alpha=0.7,
                   label=f'{methode}: {valeur:.6f}')
        plt.plot(valeur, 0, 'o', color=couleur, markersize=8)
   
    # Valeur exacte
    plt.axvline(x=valeur_exacte, color='black', linestyle='-', alpha=0.8,
               linewidth=2, label=f'Exacte: {valeur_exacte:.6f}')
    plt.plot(valeur_exacte, 0, 'kx', markersize=10, markeredgewidth=2)
   
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Comparaison des méthodes pour √{N}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=True)
# =============================================================================
# MENU PRINCIPAL
# =============================================================================
def menu_principal():
    """Menu principal interactif"""
    while True:
        print("\n" + "="*70)
        print("MÉTHODES NUMÉRIQUES AVEC SCIPY/NUMPY")
        print("="*70)
        print("1. Dichotomie (SciPy bisect) - Méthode robuste")
        print("2. Balayage (NumPy) - Méthode simple")
        print("3. Newton-Raphson (SciPy newton) - Convergence rapide")
        print("4. Sécante (SciPy newton sans dérivée) - Alternative à Newton")
        print("5. Comparaison visuelle des 4 méthodes")
        print("6. Quitter")
        print("-" * 70)
       
        choix = input("Choisissez une méthode (1-6) : ").strip()
       
        if choix == '1':
            dichotomie_scipy()
        elif choix == '2':
            balayage_numpy()
        elif choix == '3':
            newton_scipy()
        elif choix == '4':
            secante_scipy()
        elif choix == '5':
            comparer_methodes_visuelle()
        elif choix == '6':
            print("\n🎯 Programme terminé. Au revoir !")
            break
        else:
            print("❌ Choix invalide. Veuillez choisir un nombre entre 1 et 6.")
       
        input("\n↵ Appuyez sur Entrée pour continuer...")
# =============================================================================
# EXÉCUTION
# =============================================================================
if __name__ == "__main__":
    print("🔬 MÉTHODES NUMÉRIQUES POUR √N AVEC SCIPY/NUMPY")
    print("📚 Méthodes implémentées : Dichotomie, Balayage, Newton, Sécante")
    print("💡 Utilisez les fonctions natives de SciPy/NumPy pour des résultats optimisés")
    menu_principal()