# Consigliere di Stazione 🇮🇹📡

*Made in Italy by I6502TR*

---

## 📻 Cos'è?

Il Consigliere di Stazione è quello che avevi in shack prima del computer: il socio esperto che guardava il K-index sul giornale e sapeva già che i 20 metri stavano per aprirsi.

Fa esattamente quello, ma in digitale. Registra i QSO, scarica i dati solari da NOAA, controlla l'attività POTA in tempo reale, e ti dice cosa fare con tutto questo. I dati del registro non vengono inviati a servizi esterni.

**Nessun cloud. Nessun abbonamento. I tuoi log restano nello shack.**

---

## 🖼️ L'interfaccia

![Dashboard del Consigliere di Stazione](docs/screenshot.png)

Tasti grandi, colori che si leggono con qualsiasi luce, layout che non richiede manuale. Se sai accendere un computer, lo usi.

---

## ✨ Cosa fa

**Quaderno di bordo** — frequenza, modo, nominativo, RST, locatore QTH, note libere. La banda la calcola da sola. Ogni contatto ha il suo timestamp. Il tuo nominativo operatore si imposta cliccando direttamente sulla scritta in cima alla pagina — niente file da editare.

**Dati NOAA** — K-index e Solar Flux Index aggiornati con un click. Puoi confrontare un ascolto difficile di tre giorni fa con le condizioni solari di quel momento esatto.

**Il Consigliere** — analizza localmente le statistiche dei tuoi log, prende i dati NOAA e l'attività POTA e ti dice dove e quando ascoltare. Usa per impostazione predefinita il modello SWL specializzato pubblicato su Ollama. Se Ollama non è disponibile, passa automaticamente al motore deterministico basato sui dati reali.

**Stazioni attive (POTA)** — chi sta trasmettendo adesso, su quale banda, in quale parco. Filtrabile per banda e modo.

---

## 🔒 Privacy

Non c'è un server. Non c'è un account. Non c'è niente da pagare.

I QSO finiscono in un file SQLite sul tuo disco e non vengono inclusi nelle richieste NOAA o POTA. Le uniche comunicazioni esterne dell'app sono le letture dei dati pubblici NOAA e POTA; l'inferenza AI avviene su `localhost` tramite Ollama.

---

## 🚀 Installazione

### Windows

1. Scarica **[ConsigliereDiStazione.exe](https://github.com/111blackeagle111/consigliere-di-stazione/raw/main/ConsigliereDiStazione.exe)**
2. Mettilo dove vuoi (Desktop, Documenti...)
3. Doppio click
4. Il browser si apre da solo su `http://localhost:8080`

Python non serve. Non bisogna installare niente.

Per fermare il programma, chiudi la finestra nera che rimane aperta.

Prima volta? Leggi la **[Guida rapida per Windows](docs/GUIDA_WINDOWS.txt)** — scarica, Windows Defender, primo avvio, backup dei log.

### Linux / Raspberry Pi

```bash
git clone https://github.com/111blackeagle111/consigliere-di-stazione.git
cd consigliere-di-stazione
pip install -r requirements.txt
python src/main.py
```

Se vuoi il Consigliere AI con Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull amaccafeo/swlbot:latest
```

🇮🇹 **Modello consigliato:** [`amaccafeo/swlbot`](https://ollama.com/amaccafeo/swlbot) — modello fine-tuned in italiano, ottimizzato per il dominio radioamatoriale (Qwen2 3.1B, Q4_K_M, ~1.9 GB).

📋 **Requisiti:**
- **RAM:** 4 GB minimi, 8 GB consigliati
- **Disco:** ~2 GB liberi
- **Testato su:** PC x86_64, Raspberry Pi 4 (4 GB+) e Raspberry Pi 5

Il modello specializzato è quello usato automaticamente dall'app. La richiesta imposta un contesto operativo di 4.096 token, configurabile tramite `OLLAMA_NUM_CTX`.

In alternativa, puoi usare un modello generico (non specializzato):

```bash
ollama pull llama3.2:3b
OLLAMA_MODEL=llama3.2:3b python src/main.py
```

Le variabili disponibili sono:

- `OLLAMA_MODEL`: modello da usare;
- `OLLAMA_URL`: endpoint `/api/generate` di Ollama;
- `OLLAMA_NUM_CTX`: contesto effettivo della richiesta, `4096` per impostazione predefinita;
- `OLLAMA_TIMEOUT`: timeout in secondi, `60` per impostazione predefinita;
- `CONSIGLIERE_DATA_DIR`: directory personalizzata per `swl_logs.db`.

Senza Ollama funziona lo stesso: l'interfaccia segnala chiaramente che il consiglio è stato calcolato dal motore locale a regole.

## 💾 Database e architettura

SQLite è gestito tramite SQLAlchemy. Durante lo sviluppo il database rimane nella directory del progetto. Le nuove installazioni Windows salvano i dati in `%LOCALAPPDATA%\ConsigliereDiStazione\swl_logs.db`; Linux usa `${XDG_DATA_HOME:-~/.local/share}/consigliere-di-stazione/swl_logs.db`.

Per non nascondere i registri esistenti, l'eseguibile continua a usare automaticamente un vecchio `swl_logs.db` trovato accanto al file `.exe`. Puoi quindi aggiornare senza spostare subito il database.

## 📖 Press / Article

[I Built an AI Ham Radio Assistant That Runs Entirely Offline](https://medium.com/@andrea.maccafeo/i-built-an-ai-ham-radio-assistant-that-runs-entirely-offline-because-your-qso-log-is-none-of-992cac4230cf) — Medium
