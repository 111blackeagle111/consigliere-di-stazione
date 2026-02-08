# Consigliere di Stazione 🇮🇹📡

**Il registro di bordo radioamatoriale che osserva la propagazione e consiglia.**

*Made in Italy by I6502TR*

---

## 📻 Cos'è?

Il **Consigliere di Stazione** è il tuo nuovo "socio anziano" digitale: quel collega esperto che una volta stava nello shack con il quaderno di appunti e sapeva sempre quando si aprivano le bande.

Oggi fa lo stesso lavoro, ma in forma digitale e **restando fisicamente nel tuo computer**: registra i tuoi QSO, controlla automaticamente i dati solari NOAA (K-index e Solar Flux), e ti avvisa con consigli personalizzati quando le condizioni sono favorevoli.

**Nessun dato in cloud. Nessun abbonamento. Nessuna connessione obbligatoria.**

---


## 🖼️ L'interfaccia

![Dashboard del Consigliere di Stazione](docs/screenshot.png)

L'aspetto è volutamente pulito e familiare: sembra un programma che hai sempre usato. Caratteri grandi, contrasto elevato, pulsanti evidenti. Ottimizzato per l'uso in shack, anche con luce variabile.

---

## ✨ Cosa fa esattamente

### 📖 Il Quaderno di Bordo Digitale
- Registra frequenza, modo, nominativo, RST, locatore QTH e note
- Calcola automaticamente la banda (dai 160m ai 6m)
- Timestamp preciso per ogni contatto
- Note espandibili per annotare QSB, QRM, condizioni di ricezione

### 🌍 Radar Propagazione (NOAA)
- Scarica automaticamente K-index e Solar Flux Index dai server NOAA
- Mostra l'andamento della situazione geofisica in tempo reale
- Storico consultabile: confronta i tuoi ricevimenti con le condizioni solari di quel momento

### 🧠 Il Consigliere (AI Locale)
- Analizza i tuoi log storici + dati NOAA + attività POTA
- Genera avvisi intelligenti: *"Condizioni eccellenti per i 20 metri verso Est"*
- **Funziona senza internet**: l'intelligenza artificiale (modello Llama 3.2) gira sul tuo PC, non manda i tuoi dati a nessuno

### 📡 Ricerca Stazioni Attive
- Interroga il database POTA (Parks on the Air) per vedere chi trasmette in tempo reale
- Filtra per banda e modo (FT8, CW, SSB)

---

## 🔒 Privacy Totale: Zero Cloud

Contrariamente a molti logger moderni:
- ❌ I tuoi QSO **non** vanno su server remoti
- ❌ Non serve abbonamento mensile
- ❌ Non serve connessione internet permanente (solo per aggiornare i dati NOAA, se vuoi)
- ✅ Tutto resta in un file nel tuo computer (SQLite)
- ✅ L'AI lavora in locale, non usa ChatGPT o servizi esterni

**I tuoi log sono tuoi e restano nello shack.**

---

## 🚀 Installazione

### Per Windows (Raccomandato)
1. **SCARICA**: Vai nella sezione e scarica `ConsigliereDiStazione-v1.0.zip`
2. **SCOMPATTA**: Estrai la cartella dove vuoi (Desktop, Documenti, ecc.)
3. **AVVIA**: Doppio click su `ConsigliereDiStazione.exe`
4. **USA**: Si apre il browser automaticamente, oppure vai su `http://localhost:8080`

*Non serve installare Python. Non serve smanettare.*

### Per Linux / Raspberry Pi
```bash
# Clona il repository
git clone https://github.com/tuousername/consigliere-di-stazione.git
cd consigliere-di-stazione

# Installa dipendenze
pip install -r requirements.txt

# Se vuoi usare il Consigliere AI, installa anche Ollama:
# curl -fsSL https://ollama.com/install.sh | sh
# ollama pull llama3.2

# Avvia
python main.py
