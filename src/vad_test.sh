#!/bin/bash
set -e # Arrête le script en cas d'erreur

# --- Configuration ---
# Le nom de ton script Python
PYTHON_SCRIPT="vad.py"
# Le fichier audio à analyser
INPUT_WAV="test.wav"
# Le dossier où sauvegarder les graphiques
BASE_OUTPUT_DIR="vad_results"
# ---------------------

# Vérifications préliminaires
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERREUR: $PYTHON_SCRIPT introuvable."
    exit 1
fi
if [ ! -f "$INPUT_WAV" ]; then
    echo "ERREUR: $INPUT_WAV introuvable. Avez-vous lancé ffmpeg ?"
    exit 1
fi

# Fonction utilitaire pour lancer un test avec des logs clairs
run_test() {
    local test_name="$1"
    local description="$2"
    local agg="$3"
    local silence="$4"
    local min_dur="$5"
    local output_dir="$BASE_OUTPUT_DIR/$test_name"

    echo -e "\n\033[1;34m[TEST] Lancement de '$test_name'...\033[0m"
    echo " -> Description : $description"
    echo " -> Paramètres  : Agressivité=$agg | Silence=$silence ms | Durée Min=$min_dur ms"
    echo " -> Sortie      : $output_dir/"

    # Nettoyage préalable si le dossier existe
    rm -rf "$output_dir"

    # Exécution du script Python
    python3 "$PYTHON_SCRIPT" "$INPUT_WAV" \
        --output-dir "$output_dir" \
        --aggressiveness "$agg" \
        --silence-ms "$silence" \
        --min-duration-ms "$min_dur"

    echo -e "\033[1;32m[OK] Test '$test_name' terminé.\033[0m"
}

# ==============================================================================
# CAMPAGNE DE TESTS
# ==============================================================================

echo "=== DÉBUT DE LA CAMPAGNE DE TESTS VAD ==="
rm -rf "$BASE_OUTPUT_DIR" # On repart de zéro pour éviter les mélanges
mkdir -p "$BASE_OUTPUT_DIR"


# TEST 1 : Standard
# C'est votre point de départ. Bon équilibre pour un discours posé.
run_test "01_Standard" \
    "Réglage équilibré pour discours normal." \
    3 500 10000

# TEST 2 : Haute Sensibilité (Discours rapide)
# Réduit le temps de silence nécessaire pour couper. Utile si le locuteur
# enchaîne très vite sans respirer. Risque de couper des fins de mots si trop bas.
run_test "02_Sensible_FastTalker" \
    "Pour locuteurs rapides : coupe dès 300ms de pause." \
    3 300 10000

# TEST 3 : Segmentation "Paragraphe"
# Ne coupe que sur les longues pauses. Utile si on veut donner des gros blocs
# aux sous-titreurs plutôt que phrase par phrase.
run_test "03_Macro_Segments" \
    "Gros blocs : ne coupe qu'après 1.5s de silence." \
    3 1500 10000

# TEST 4 : Debug Bruit (Agressivité 0)
# Permet de voir si votre audio est considéré comme bruyant.
# Si ce test ne produit qu'un seul gros segment alors que le TEST 1 fonctionne,
# c'est que votre VAD filtre efficacement un bruit de fond constant.
run_test "04_Debug_NoFilter" \
    "Agressivité 0 : très sensible au bruit de fond." \
    0 500 10000

echo -e "\n\033[1;32m=== VAD TEST TERMINÉE AVEC SUCCÈS ===\033[0m"
echo "Explorez le dossier '$BASE_OUTPUT_DIR' pour voir et ÉCOUTER les résultats."