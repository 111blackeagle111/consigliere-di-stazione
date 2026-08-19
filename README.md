# Consigliere di Stazione 🇮🇹📡

*Made in Italy by I6502TR*

---

## 📻 Cos'è?

Il Consigliere di Stazione è quello che avevi in shack prima del computer: il socio esperto che guardava il K-index sul giornale e valutava se valeva la pena provare i 20 metri.

Fa esattamente quello, ma in digitale. Registra i QSO, scarica i dati solari da NOAA, controlla l'attività POTA in tempo reale su tutte le bande e combina questi dati con le statistiche del tuo intero storico operativo. I dati del registro non vengono inviati a NOAA o POTA.

**Nessun cloud per il registro. Nessun abbonamento. I tuoi log restano nello shack.**

---

## 🖼️ L'interfaccia

![Dashboard del Consigliere di Stazione](docs/screenshot.png)

Tasti grandi, colori che si leggono con qualsiasi luce, layout che non richiede manuale. Se sai accendere un computer, lo usi.

Lo storico completo viene caricato soltanto quando serve, con ricerca, filtri per banda e modo e 50 QSO per pagina:

![Storico completo QSO con filtri](docs/history.png)

*Le schermate usano dati sintetici marcati `[DEMO-6M]`; non rappresentano collegamenti reali.*

---

## ✨ Cosa fa

**Quaderno di bordo** — frequenza, modo, nominativo, RST, locatore QTH, note libere. La banda la calcola da sola. Ogni contatto ha il suo timestamp. Nominativo e locator Maidenhead dell'operatore si impostano cliccando direttamente sulle scritte in cima alla pagina — niente file da editare.

**Storico completo** — gli ultimi dieci QSO restano visibili nella dashboard; l'archivio completo si apre su richiesta e permette di cercare nominativi, locator e note, filtrare banda e modo, scorrere 50 record per pagina, modificare ed eliminare i QSO manuali. I record tecnici NOAA e gli alert non vengono mescolati ai QSO.

**Esportazioni e backup** — dalla dashboard puoi scaricare tutti i QSO manuali in CSV o ADIF e creare una copia coerente del database SQLite anche mentre l'app è aperta. Le date restano locali nell'interfaccia; il CSV riporta sia ora locale sia UTC, mentre ADIF usa UTC.

**Dati NOAA** — K-index e Solar Flux Index aggiornati con un click. Ogni nuovo aggiornamento manuale viene conservato nel database come record tecnico, separato dai QSO visibili. NOAA e POTA usano cache locali di 60 e 30 secondi; l'interfaccia indica ora del dato, età e stato della cache e può mostrare l'ultimo dato valido se il servizio cade.

**Il Consigliere** — include **tutti** i QSO manuali, senza limiti temporali o numerici, facendo calcolare a SQLite i conteggi per banda e modo senza caricare l'intero registro in memoria. Queste statistiche aggregate vengono unite a K-index, SFI e attività POTA reale su tutte le bande. Usa il servizio locale swlbot RAG, con fallback a Qwen diretto e infine alle regole locali.

> **Work in progress:** l'integrazione con il modello linguistico locale dell'autore è ancora sperimentale e in fase di sviluppo. I consigli generati dall'IA vanno quindi verificati confrontandoli con i dati NOAA, POTA e con la propria esperienza operativa.

**Stazioni attive (POTA)** — mostra nominativo, frequenza, parco e locator degli spot correnti. Se hai impostato il QTH, calcola localmente distanza, azimut e direzione per gli spot dotati di locator. La ricerca è filtrabile per banda e modo e visualizza fino a 10 risultati.

### I due controlli AI

- **Controlla Ora (Genera Alert)** ottiene NOAA e POTA su tutte le bande, riutilizzando la cache breve quando valida, usa il QTH per interpretare gli spot localizzati e valuta gli indicatori operativi correnti. Se ci sono elementi sufficienti genera e salva un alert tecnico; non usa lo storico QSO.
- **Chiedi Consiglio all'IA (usa anche il tuo storico)** ottiene NOAA e POTA su tutte le bande, usa le distanze calcolate dal QTH e le statistiche aggregate di tutti i QSO manuali. Il consiglio non viene salvato come QSO.

Il percorso di generazione è sempre locale:

1. **swlbot RAG + modello locale** con il corpus tecnico;
2. se il RAG non risponde, **Qwen diretto** vincolato alla valutazione deterministica, con un avviso visibile;
3. se anche Ollama non risponde, **regole locali**;
4. se una risposta AI contiene elementi non supportati, viene scartata: si passa al livello successivo oppure direttamente alle **regole locali verificate**.

Una barriera deterministica controlla inoltre le risposte di RAG e Qwen: frequenze non presenti nei dati live e deduzioni non supportate su MUF, rumore, DX o aperture vengono scartate. Se il RAG risponde ma non supera questa verifica, l'interfaccia lo dichiara e usa Qwen soltanto per riformulare le regole verificate.

---

## 🔒 Privacy

Non c'è un server remoto dell'applicazione. Non c'è un account. Non c'è niente da pagare.

I QSO finiscono in un file SQLite sul tuo disco e non vengono inclusi nelle richieste NOAA o POTA. Le uniche comunicazioni esterne predefinite dell'app sono le letture dei dati pubblici NOAA e POTA; l'inferenza AI avviene su `localhost` tramite swlbot RAG o Ollama.

Il server dell'app ascolta soltanto su `127.0.0.1` per impostazione predefinita e rifiuta le scritture provenienti da origini web diverse. L'accesso dalla rete locale è un'opzione avanzata: imposta `CONSIGLIERE_HOST=0.0.0.0` solo su una rete fidata, perché l'app non implementa account o autenticazione.

Se configuri `SWLBOT_RAG_URL` o `DIRECT_LLM_URL` verso un computer remoto, il riepilogo operativo inviato a quell'endpoint può comprendere QTH, statistiche per banda e modo e dati NOAA/POTA. I singoli record QSO non vengono inclusi nel prompt attuale.

---

## 🚀 Installazione

### Windows

1. Apri la pagina **[Releases](https://github.com/111blackeagle111/consigliere-di-stazione/releases)** e scarica `ConsigliereDiStazione.exe` dalla versione più recente
2. Mettilo dove vuoi (Desktop, Documenti...)
3. Doppio click
4. Il browser si apre da solo su `http://localhost:8080`

Per registro, NOAA, POTA e consigli deterministici non serve Python e non bisogna installare altro. Il modello linguistico non è incluso nell'eseguibile: per i consigli generati da RAG o Qwen occorrono Ollama e i servizi locali descritti sotto.

Per fermare il programma, chiudi la finestra nera che rimane aperta.

Prima volta? Leggi la **[Guida rapida per Windows](docs/GUIDA_WINDOWS.txt)** — download, verifica SHA-256, Windows Defender, primo avvio e backup dei log.

### Linux / Raspberry Pi

```bash
git clone https://github.com/111blackeagle111/consigliere-di-stazione.git
cd consigliere-di-stazione
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python src/main.py
```

Per usare i consigli generati dal modello linguistico, installa e avvia Ollama con il modello configurato. Per aggiungere anche il corpus tecnico, avvia swlbot RAG sulla porta 8081:

```bash
cd ../swl-rag
python server.py
```

In un secondo terminale avvia l'applicazione:

```bash
cd consigliere-di-stazione
.venv/bin/python src/main.py
```

📋 **Requisiti per le funzioni AI facoltative:**
- **RAM:** 4 GB minimi, 8 GB consigliati per modello e RAG
- **Disco:** spazio sufficiente per Ollama, il modello e l'indice ChromaDB
- **Testato su:** PC x86_64, Raspberry Pi 4 (4 GB+) e Raspberry Pi 5

Le variabili disponibili sono:

- `SWLBOT_RAG_URL`: endpoint del servizio, predefinito `http://127.0.0.1:8081/api/advice`;
- `SWLBOT_RAG_TIMEOUT`: timeout in secondi, `90` per impostazione predefinita;
- `DIRECT_LLM_URL`: endpoint Ollama diretto, predefinito `http://127.0.0.1:11434/api/chat`;
- `DIRECT_LLM_MODEL`: modello di fallback, predefinito `qwen3.5:4b`;
- `DIRECT_LLM_TIMEOUT`: timeout del fallback diretto, `90` secondi;
- `CONSIGLIERE_DATA_DIR`: directory personalizzata per `swl_logs.db`;
- `CONSIGLIERE_HOST`: indirizzo di ascolto, predefinito `127.0.0.1`;
- `CONSIGLIERE_PORT`: porta locale, predefinita `8080`.

Senza swlbot RAG l'app prova Qwen direttamente e mostra **CONSIGLIO QWEN SENZA RAG**. Se anche Ollama non è disponibile, mostra **CONSIGLIO LOCALE** calcolato con le regole. L'integrazione con il modello locale è ancora sperimentale e non fa parte dell'eseguibile autonomo.

## 💾 Database e architettura

SQLite è gestito tramite SQLAlchemy. Durante lo sviluppo il database rimane nella directory del progetto. Le nuove installazioni Windows salvano i dati in `%LOCALAPPDATA%\ConsigliereDiStazione\swl_logs.db`; Linux usa `${XDG_DATA_HOME:-~/.local/share}/consigliere-di-stazione/swl_logs.db`.

Per non nascondere i registri esistenti, l'eseguibile continua a usare automaticamente un vecchio `swl_logs.db` trovato accanto al file `.exe`. Puoi quindi aggiornare senza spostare subito il database.

Dalla pagina principale sono disponibili **Esporta CSV**, **Esporta ADIF** e **Backup database**. CSV e ADIF includono soltanto i QSO manuali; il backup conserva anche impostazioni e record tecnici.

Per compatibilità, i timestamp già presenti nel database restano nel formato locale storico: l'interfaccia li mostra in ora locale, il CSV aggiunge la corrispondente colonna UTC e l'ADIF viene prodotto in UTC. Non viene eseguita una migrazione distruttiva dei vecchi orari.

## 🏷️ Versioni e release

Lo stato della prossima versione è in [CHANGELOG.md](CHANGELOG.md). Una release viene creata soltanto da un tag `vX.Y.Z`: la pipeline controlla che il tag corrisponda al file `VERSION`, compila l'EXE su Windows e pubblica anche il checksum SHA-256. Finché `VERSION` termina in `-dev`, la pubblicazione è bloccata intenzionalmente.

Per eseguire la suite di sviluppo:

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
```

### Generare uno storico dimostrativo

Per test e screenshot puoi inserire 240 QSO sintetici distribuiti sugli ultimi sei mesi:

```bash
.venv/bin/python scripts/seed_demo_qsos.py
```

Lo script crea prima un backup del database e marca ogni record con `[DEMO-6M]`. Se i dati demo sono già presenti, non li duplica; usa `--replace-demo` soltanto per rigenerarli.

## 📖 Press / Article

[I Built an AI Ham Radio Assistant That Runs Entirely Offline](https://medium.com/@andrea.maccafeo/i-built-an-ai-ham-radio-assistant-that-runs-entirely-offline-because-your-qso-log-is-none-of-992cac4230cf) — Medium
