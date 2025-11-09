# Module de Segmentation Audio VAD (Projet STC)

## 1\. Objectif

Ce module implémente une stratégie de fragmentation audio basée sur la **Détection d'Activité Vocale (VAD)**. Son but est d'analyser un flux audio (simulé à partir d'un fichier `.wav`) et de le découper en segments sémantiquement cohérents.

Le principe est de détecter les **silences naturels** dans un discours pour marquer la fin d'une séquence, plutôt que de couper arbitrairement à des intervalles de temps fixes.

Ce dépôt fournit les outils pour :

  * **Segmenter** un fichier audio en utilisant la bibliothèque `webrtcvad`.
  * **Visualiser** les segments produits sur la forme d'onde du signal et générer des graphiques PNG.
  * **Sauvegarder** les segments audio résultants sous forme de fichiers `.wav` individuels pour une vérification auditive.

## 2\. Prérequis et Installation

### 2.1. Dépendances Système

Cet outil nécessite `python3`, `pip`, `ffmpeg`, et les outils de compilation Python.

```bash
# Sur Debian/Ubuntu
sudo apt update
sudo apt install python3 python3-pip python3-venv ffmpeg build-essential
```

### 2.2. Environnement Python

```bash
# 1. Créez l'environnement dans le dossier de votre projet
python3 -m venv venv

# 2. Activez-le (vous devrez le faire à chaque nouvelle session de terminal)
source venv/bin/activate

# 3. Installez les dépendances Python
pip install webrtcvad matplotlib numpy setuptools
```

### 2.3. Préparation du Fichier Audio

Le VAD est très strict sur le format audio. Votre fichier d'entrée **doit** être :

  * **WAV** (non compressé)
  * **16000 Hz** (Taux d'échantillonnage)
  * **16-bit** (Profondeur)
  * **Mono** (1 seul canal)

Utilisez `ffmpeg` pour convertir n'importe quel fichier audio (mp3, m4a, etc.) en ce format exact.

```bash
# Exemple de conversion
ffmpeg -i votre_discours_source.mp3 -ar 16000 -ac 1 -acodec pcm_s16le test.wav
```

## 3\. Utilisation de `vad.py`

Le script `vad.py` est l'outil principal. Il ne peut pas être lancé seul.

**Usage Minimal :**

```bash
python3 vad.py test.wav
```

Cela exécutera le script sur `test.wav` avec tous les paramètres par défaut et créera un dossier `output/` avec les résultats.

### 3.2. Options de Ligne de Commande

Pour une analyse fine, vous pouvez utiliser les arguments suivants :

| Argument | Raccourci | Défaut | Description |
| :--- | :--- | :--- | :--- |
| `wav_file` | (aucun) | (Requis) | Le fichier `.wav` (16kHz, mono, 16-bit) à analyser. |
| `--output-dir` | `-o` | `output` | Dossier où sauvegarder le graphique et les segments audio. |
| `--aggressiveness` | `-a` | `3` | Agressivité du VAD (0, 1, 2, ou 3). |
| `--silence-ms` | `-s` | `500` | Durée minimale du silence (en ms) pour déclencher une coupure. |
| `--min-duration-ms` | `-d` | `1000` | Durée minimale (en ms) d'un segment de parole pour être conservé. |

**Exemple d'utilisation avancée :**

```bash
python3 vad.py test.wav -o "mon_test" -a 3 -s 400 -d 800
```

-----

## 4\. Analyse des Paramètres

Les 3 paramètres (`-a`, `-s`, `-d`) interagissent. Voici comment les régler.

### 4.1. Agressivité (`-a` / `--aggressiveness`)

  * **Signification Directe :** Le "niveau de confiance" du VAD (de 0 à 3) pour classer un chunk audio comme "Voix".
  * **Impact Indirect  :** C'est le **filtre anti-bruit**.
  * **Cas d'usage :**
      * **`-a 0` (Très sensible) :** Le VAD entend *tout*. Le moindre bruit de fond (climatisation, sifflement de micro, respiration forte) sera classé comme "Voix".
      * **`-a 3` (Très strict) :** Le VAD n'accepte que les signaux clairs comme de la "Voix" et classe les bruits de fond faibles comme du "Silence".

### 4.2. Silence Minimum (`-s` / `--silence-ms`)

  * **Signification Directe :** La durée (en ms) de silence *continu* que le VAD doit détecter avant de décider "OK, la phrase est finie, je coupe".
  * **Impact Indirect :** C'est le **contrôle de granularité**. Il définit ce que vous considérez comme une "pause".
  * **Cas d'usage :**
      * **`-s 1200` (Valeur élevée) :** Vous ne couperez que lors de très longues pauses (fins de paragraphe, le locuteur réfléchit). Les pauses de respiration normales seront ignorées.
      * **`-s 500` (Valeur moyenne) :** Réglage standard. Capture bien les pauses naturelles entre les phrases.
      * **`-s 300` (Valeur basse) :** Très sensible. À utiliser si le locuteur parle très vite (style "streamer") avec des pauses très courtes.

### 4.3. Durée Minimale (`-d` / `--min-duration-ms`)

**Signification directe :** Règle de segmentation — le script ne déclenche pas une coupure à un silence même long tant que le segment en cours n'a pas atteint cette durée minimale.

**Impact indirect :** Contrôle de "fusion" qui évite les micro-segments. Le silence est conservé dans le segment courant jusqu'à ce que sa durée totale dépasse `-d`, ce qui attache les petits mots (ex. `OK`, `Euh`) à la phrase suivante.

**Cas d'usage :**
- `-d 0` (désactivé) : mode "brut" — coupe à chaque silence valide, produisant de nombreux petits segments.
- `-d 1500` (1,5 s) : bon équilibre — les interjections courtes sont rattachées à la phrase suivante.
- `-d 5000` (5 s) : gros blocs — fusionne plusieurs courtes phrases en segments longs (sauf en fin de fichier).

**Stratégie de réglage :**
- Commencez bas (par ex. `-d 500`) pour voir toutes les coupures, puis augmentez si vous obtenez trop de segments trop courts.
- Ajustez `-d` en conjonction avec `--silence-ms` (`-s`) et `--aggressiveness` (`-a`) pour affiner la granularité.

Écoutez les fichiers audio. Si vous avez trop de segments très courts qui devraient logiquement être avec la phrase suivante, augmentez cette valeur.
## 5\. Campagne de Tests (`run_campaign.sh`)

Pour faciliter ces tests, le script `run_campaign.sh` automatise l'exécution de `vad.py` avec 5 combinaisons de paramètres stratégiques.

**Rendre le script exécutable (une seule fois) :**

```bash
chmod +x run_campaign.sh
```

**Lancer la campagne de tests :**

```bash
./run_campaign.sh
```

### 5.1. Scénarios de Test

Le script créera un dossier `campaign_results/` avec les sous-dossiers suivants :

1.  **`01_Standard` (`-a 3 -s 500 -d 1000`) :**

      * **Cas d'usage :** Le point de départ équilibré. Idéal pour un discours clair et posé.

2.  **`02_Sensible_FastTalker` (`-a 3 -s 300 -d 800`) :**

      * **Cas d'usage :** Pour un locuteur rapide. La réduction du silence (`-s 300`) permet de capter les pauses plus courtes entre les phrases.

3.  **`03_Macro_Segments` (`-a 3 -s 1500 -d 2000`) :**

      * **Cas d'usage :** Segmentation par "paragraphe". En ignorant les silences de moins de 1.5s, on ne garde que les grosses pauses, créant des segments plus longs.

4.  **`04_Debug_NoFilter` (`-a 0 -s 500 -d 1000`) :**

      * **Cas d'usage :** Diagnostic de bruit. Si ce test produit 1 seul segment géant alors que `01_Standard` en produit plusieurs, cela prouve que votre réglage d'agressivité (`-a 3`) est essentiel et fonctionne bien.

5.  **`05_Raw_AllSegments` (`-a 3 -s 400 -d 0`) :**

      * **Cas d'usage :** Analyse "brute". En mettant la durée minimale à 0, vous verrez *tout* ce que le VAD considère comme de la parole, y compris les déchets. Utile pour comprendre ce que le filtre `-d` nettoie.

### 5.2. Interprétation des Résultats

Pour chaque dossier de test, l'analyse se fait en deux temps :

1.  **Visuellement (`analyse_resultat.png`) :**

      * Regardez le graphique du haut (le signal). Repérez les "vagues" (parole) et les "plats" (silence).
      * Regardez le graphique du bas (les segments). Les barres de couleur (`#1`, `#2`...) s'alignent-elles logiquement avec les vagues ? Les espaces entre les barres correspondent-ils aux zones plates ?

2.  **Auditivement (Dossier `audio/`) :**

      * **C'est la seule vérité.** Écoutez les fichiers `seg_001.wav`, `seg_002.wav`, etc., dans l'ordre.
      * La coupure est-elle nette ?