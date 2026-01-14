#!/bin/bash
#
# Clarity - One-Click Installer for macOS
# Quartalsabweichungen verstehen. In Sekunden statt Stunden.
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

clear
echo ""
echo -e "${BLUE}${BOLD}"
echo "   ██████╗██╗      █████╗ ██████╗ ██╗████████╗██╗   ██╗"
echo "  ██╔════╝██║     ██╔══██╗██╔══██╗██║╚══██╔══╝╚██╗ ██╔╝"
echo "  ██║     ██║     ███████║██████╔╝██║   ██║    ╚████╔╝ "
echo "  ██║     ██║     ██╔══██║██╔══██╗██║   ██║     ╚██╔╝  "
echo "  ╚██████╗███████╗██║  ██║██║  ██║██║   ██║      ██║   "
echo "   ╚═════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝   ╚═╝      ╚═╝   "
echo -e "${NC}"
echo -e "${BOLD}  Quartalsabweichungen verstehen. In Sekunden statt Stunden.${NC}"
echo ""
echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Function to print status
status() {
    echo -e "${BLUE}[*]${NC} $1"
}

success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    error "Dieses Installationsprogramm ist nur für macOS."
    exit 1
fi

# ============================================
# STEP 1: Check/Install Homebrew
# ============================================
echo -e "${BOLD}Schritt 1/5: Systemvoraussetzungen prüfen${NC}"
echo ""

if ! command -v brew &> /dev/null; then
    status "Homebrew wird installiert (macOS Paketmanager)..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    # Add Homebrew to PATH for Apple Silicon
    if [[ -f "/opt/homebrew/bin/brew" ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
    success "Homebrew installiert"
else
    success "Homebrew bereits vorhanden"
fi

# ============================================
# STEP 2: Check/Install Python
# ============================================
echo ""
echo -e "${BOLD}Schritt 2/5: Python prüfen${NC}"
echo ""

if ! command -v python3 &> /dev/null; then
    status "Python 3 wird installiert..."
    brew install python@3.11
    success "Python installiert"
else
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    success "Python $PYTHON_VERSION bereits vorhanden"
fi

# ============================================
# STEP 3: Check/Install Ollama
# ============================================
echo ""
echo -e "${BOLD}Schritt 3/5: KI-Engine (Ollama) installieren${NC}"
echo ""

if ! command -v ollama &> /dev/null; then
    status "Ollama wird installiert..."
    brew install ollama
    success "Ollama installiert"
else
    success "Ollama bereits vorhanden"
fi

# ============================================
# STEP 4: Download AI Model
# ============================================
echo ""
echo -e "${BOLD}Schritt 4/5: KI-Modell herunterladen${NC}"
echo ""

# Start Ollama in background if not running
if ! pgrep -x "ollama" > /dev/null; then
    status "Ollama wird gestartet..."
    ollama serve &> /dev/null &
    sleep 3
fi

# Check if model exists
if ollama list 2>/dev/null | grep -q "llama3.2"; then
    success "KI-Modell (llama3.2) bereits vorhanden"
else
    status "KI-Modell wird heruntergeladen (ca. 2 GB)..."
    echo -e "${YELLOW}    Dies kann einige Minuten dauern...${NC}"
    ollama pull llama3.2
    success "KI-Modell heruntergeladen"
fi

# ============================================
# STEP 5: Setup Python Environment
# ============================================
echo ""
echo -e "${BOLD}Schritt 5/5: Clarity einrichten${NC}"
echo ""

# Create virtual environment if not exists
if [ ! -d ".venv" ]; then
    status "Python-Umgebung wird erstellt..."
    python3 -m venv .venv
    success "Python-Umgebung erstellt"
else
    success "Python-Umgebung bereits vorhanden"
fi

# Activate and install dependencies
status "Abhängigkeiten werden installiert..."
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
success "Abhängigkeiten installiert"

# ============================================
# Create Launcher Script
# ============================================
cat > "$SCRIPT_DIR/Clarity_Starten.command" << 'LAUNCHER'
#!/bin/bash
#
# Clarity - Launcher
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'
BOLD='\033[1m'

clear
echo ""
echo -e "${BLUE}${BOLD}Clarity wird gestartet...${NC}"
echo ""

# Start Ollama if not running
if ! pgrep -x "ollama" > /dev/null; then
    echo -e "${GREEN}[*]${NC} KI-Engine wird gestartet..."
    ollama serve &> /dev/null &
    sleep 2
fi

# Activate virtual environment
source .venv/bin/activate

# Start Streamlit
echo -e "${GREEN}[*]${NC} App wird gestartet..."
echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  Browser öffnet sich automatisch."
echo -e "  Falls nicht: ${BLUE}http://localhost:8501${NC}"
echo ""
echo -e "  ${BOLD}Zum Beenden:${NC} Dieses Fenster schließen oder Ctrl+C"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

streamlit run streamlit_app.py --server.headless false
LAUNCHER

chmod +x "$SCRIPT_DIR/Clarity_Starten.command"

# ============================================
# Done!
# ============================================
echo ""
echo -e "${GREEN}${BOLD}"
echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✓ INSTALLATION ABGESCHLOSSEN!"
echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${NC}"
echo ""
echo -e "  ${BOLD}So starten Sie Clarity:${NC}"
echo ""
echo -e "  1. Doppelklick auf ${BLUE}Clarity_Starten.command${NC}"
echo ""
echo -e "  2. Browser öffnet sich automatisch"
echo ""
echo -e "  ${BOLD}Hinweis:${NC} Ihre Daten bleiben 100% auf diesem Rechner."
echo "           Keine Cloud, keine Internetverbindung nötig."
echo ""
echo -e "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Ask if user wants to start now
echo -e "  ${BOLD}Möchten Sie Clarity jetzt starten? (j/n)${NC}"
read -n 1 -r
echo ""

if [[ $REPLY =~ ^[Jj]$ ]]; then
    echo ""
    exec "$SCRIPT_DIR/Clarity_Starten.command"
fi
