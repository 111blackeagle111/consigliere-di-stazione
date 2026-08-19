# Consigliere di Stazione 🇮🇹📡

*Made in Italy by I6502TR*

Il Consigliere di Stazione prova a riportare nello shack una figura familiare: il socio esperto che dà un'occhiata al K-index, ascolta cosa succede sulle bande e ti dice se vale la pena fare un tentativo.

L'applicazione tiene il registro dei QSO, legge i dati solari NOAA e mostra l'attività POTA corrente. Quando chiedi un consiglio, mette insieme questi dati con le statistiche del tuo storico. Il registro rimane in un database SQLite sul tuo computer e non viene inviato a NOAA o POTA.

Non richiede account o abbonamenti. Per usare il registro, consultare NOAA e POTA o ricevere i consigli basati sulle regole non serve alcun servizio cloud.

## Interfaccia

![Dashboard del Consigliere di Stazione](docs/screenshot.png)

La dashboard raccoglie il quaderno di bordo e i dati operativi senza nascondere le funzioni principali nei menu. Gli ultimi dieci QSO rimangono subito visibili; il resto del registro si apre quando serve.

![Storico completo QSO con filtri](docs/history.png)

Nello storico completo puoi cercare nominativi, locator e note, filtrare per banda o modo e scorrere 50 QSO per pagina.

*Le schermate usano dati sintetici marcati `[DEMO-6M]`; non rappresentano collegamenti reali.*

## Cosa puoi fare

### Tenere il registro

Per ogni QSO puoi salvare frequenza, modo, nominativo, RST ricevuto, locator QTH e note. La banda viene ricavata automaticamente dalla frequenza. Nominativo e locator Maidenhead dell'operatore si impostano cliccando sulle rispettive scritte in cima alla pagina.

Lo storico comprende tutti i QSO inseriti a mano, senza limiti temporali o numerici. Puoi modificarli o eliminarli; i record tecnici creati dagli aggiornamenti NOAA e dagli alert restano separati e non compaiono nel registro operativo.

Dalla dashboard puoi esportare tutti i QSO manuali in CSV o ADIF. Puoi anche scaricare un backup coerente dell'intero database SQLite mentre l'applicazione è aperta.

### Consultare NOAA e POTA

Il pulsante NOAA legge K-index e Solar Flux Index. Gli aggiornamenti richiesti manualmente vengono conservati nel database come record tecnici, senza confonderli con i QSO.

La ricerca POTA mostra fino a dieci spot correnti filtrabili per banda e modo, con nominativo, frequenza, parco e locator. Se hai impostato il tuo QTH, per gli spot dotati di locator vengono calcolati localmente distanza, azimut e direzione.

Per evitare richieste duplicate, NOAA usa una cache di 60 secondi e POTA una di 30. L'interfaccia indica l'orario del dato e l'età della cache. Se uno dei servizi smette temporaneamente di rispondere, l'app può recuperare l'ultimo valore valido e lo segnala chiaramente.

### Chiedere un consiglio

I due pulsanti AI hanno compiti diversi.

- **Controlla Ora (Genera Alert)** legge NOAA e l'attività POTA su tutte le bande, usa il QTH per interpretare gli spot localizzati e valuta soltanto la situazione corrente. Se trova elementi utili, genera e salva un alert tecnico. Non consulta lo storico QSO.

- **Chiedi Consiglio all'IA (usa anche il tuo storico)** parte dagli stessi dati correnti, ma aggiunge le statistiche di tutti i QSO manuali presenti nel database. Il testo risultante viene mostrato nella pagina e non viene salvato come QSO.

Non occorre premere prima i pulsanti NOAA o POTA: entrambi i controlli raccolgono direttamente ciò che serve, riutilizzando la cache quando è ancora valida. Per lo storico, conteggi e raggruppamenti vengono calcolati da SQLite; i singoli QSO non vengono caricati tutti in memoria né inseriti nel prompt.

## Come viene generato il consiglio

L'app prova questi percorsi, nell'ordine:

1. swlbot RAG con il corpus tecnico e il modello locale;
2. Qwen tramite Ollama, senza RAG, se il primo servizio non risponde;
3. le regole deterministiche integrate, se non è disponibile alcun modello.

Le risposte di RAG e Qwen passano comunque attraverso una verifica locale. Frequenze assenti dai dati correnti e deduzioni non supportate su MUF, rumore, DX o aperture vengono scartate. In quel caso l'app passa al livello successivo e spiega nell'interfaccia cosa è successo. Se nessun testo generato supera il controllo, mostra soltanto il consiglio calcolato dalle regole.

L'integrazione con il modello linguistico locale dell'autore è ancora un lavoro in corso. Gli avvisi **CONSIGLIO QWEN SENZA RAG**, **CONSIGLIO QWEN DOPO VERIFICA RAG** e **CONSIGLIO LOCALE DOPO VERIFICA AI** servono a rendere visibile quale percorso è stato usato. Un consiglio radio resta comunque un'indicazione: va confrontato con NOAA, spot POTA e segnali realmente ricevuti.

## Privacy e accesso di rete

Per impostazione predefinita l'applicazione comunica all'esterno soltanto per leggere i dati pubblici NOAA e POTA. Il registro rimane sul disco; le richieste a questi servizi non contengono i QSO.

swlbot RAG e Ollama vengono contattati su `localhost`. Se imposti `SWLBOT_RAG_URL` o `DIRECT_LLM_URL` verso un altro computer, il riepilogo inviato a quell'indirizzo può comprendere QTH, statistiche aggregate per banda e modo e dati NOAA/POTA. I singoli record QSO non fanno parte del prompt attuale.

Il server web ascolta soltanto su `127.0.0.1` e rifiuta le scritture provenienti da origini diverse. Puoi abilitarne volontariamente l'accesso dalla rete locale con `CONSIGLIERE_HOST=0.0.0.0`, ma fallo solo su una rete fidata: l'app non gestisce utenti, password o autenticazione e non deve essere esposta direttamente a Internet.

## Installazione

### Windows

1. Apri la pagina [Releases](https://github.com/111blackeagle111/consigliere-di-stazione/releases).
2. Scarica `ConsigliereDiStazione.exe` dalla versione più recente.
3. Salvalo sul Desktop, in Documenti o in un'altra cartella a tua scelta.
4. Avvialo con un doppio clic. Il browser si aprirà su `http://localhost:8080`.

Python non è necessario. L'eseguibile comprende il registro, l'accesso a NOAA e POTA e il motore di regole. Ollama, Qwen, il corpus tecnico e swlbot RAG non sono inclusi e servono soltanto per i testi generati dal modello.

Per arrestare l'applicazione, premi Invio nella finestra nera oppure chiudila. Chiudere soltanto la scheda del browser non ferma il server.

Per il primo avvio, SmartScreen, il checksum e il backup del registro consulta la [guida rapida per Windows](docs/GUIDA_WINDOWS.txt).

### Linux e Raspberry Pi

```bash
git clone https://github.com/111blackeagle111/consigliere-di-stazione.git
cd consigliere-di-stazione
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python src/main.py
```

L'app è disponibile su `http://localhost:8080`.

### Requisiti in breve

Non è stato ancora stabilito un minimo certificato con benchmark comparabili. L'applicazione base è stata usata anche su Raspberry Pi 4 con 4 GB; l'eseguibile Windows attuale è a 64 bit e la pipeline usa Python 3.12. Il registro funziona senza Internet, mentre NOAA e POTA richiedono una connessione.

Qwen e swlbot RAG sono facoltativi. Con il modello predefinito attuale, 4 GB non sono un obiettivo realistico per l'intero sistema: 8 GB sono una base pratica da valutare e 16 GB sono consigliati quando app, modello e RAG girano sullo stesso computer. La pagina [Requisiti di sistema](docs/REQUISITI.md) separa i dati misurati dalle raccomandazioni.

### Modello locale e RAG, facoltativi

Per ottenere testi generati dal modello, installa Ollama dalla [pagina ufficiale](https://ollama.com/download), seguendo le istruzioni aggiornate per il tuo sistema operativo. Poi scarica il modello configurato:

```bash
ollama pull qwen3.5:4b
```

La procedura completa, comprese verifica dell'installazione e configurazione facoltativa del RAG, è nella pagina [Requisiti di sistema](docs/REQUISITI.md#installare-i-componenti).

Per aggiungere anche il corpus tecnico, avvia swlbot RAG sulla porta 8081:

```bash
cd ../swl-rag
python server.py
```

In un altro terminale avvia il Consigliere:

```bash
cd consigliere-di-stazione
.venv/bin/python src/main.py
```

Le variabili disponibili sono:

- `SWLBOT_RAG_URL`: endpoint RAG, predefinito `http://127.0.0.1:8081/api/advice`;
- `SWLBOT_RAG_TIMEOUT`: timeout RAG, predefinito `90` secondi;
- `DIRECT_LLM_URL`: endpoint Ollama, predefinito `http://127.0.0.1:11434/api/chat`;
- `DIRECT_LLM_MODEL`: modello diretto, predefinito `qwen3.5:4b`;
- `DIRECT_LLM_TIMEOUT`: timeout del modello diretto, predefinito `90` secondi;
- `CONSIGLIERE_DATA_DIR`: cartella personalizzata per `swl_logs.db`;
- `CONSIGLIERE_HOST`: indirizzo di ascolto, predefinito `127.0.0.1`;
- `CONSIGLIERE_PORT`: porta locale, predefinita `8080`.

Senza swlbot RAG l'app prova Qwen direttamente. Se Ollama non risponde, usa le regole locali: il registro e le altre funzioni continuano a funzionare.

## Database, aggiornamenti e orari

Durante lo sviluppo, SQLite salva `swl_logs.db` nella directory del progetto. Le nuove installazioni Windows usano `%LOCALAPPDATA%\ConsigliereDiStazione\swl_logs.db`; Linux usa `${XDG_DATA_HOME:-~/.local/share}/consigliere-di-stazione/swl_logs.db`.

Se l'eseguibile trova un vecchio `swl_logs.db` accanto al file `.exe`, continua a usarlo. In questo modo un aggiornamento non nasconde il registro già esistente.

CSV e ADIF includono soltanto i QSO manuali. Il backup SQLite conserva anche impostazioni, dati tecnici NOAA e alert. L'interfaccia mostra le date in ora locale; il CSV contiene sia l'ora locale sia UTC e l'ADIF usa UTC. I vecchi timestamp non vengono riscritti durante l'aggiornamento.

## Sviluppo e release

Lo stato della prossima versione è descritto nel [changelog](CHANGELOG.md). Una release viene pubblicata soltanto da un tag `vX.Y.Z` coerente con il file `VERSION`. Se `VERSION` termina in `-dev`, la pubblicazione viene bloccata. La pipeline Windows genera l'eseguibile e il relativo checksum SHA-256.

Per eseguire i test:

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
```

Per popolare il registro con 240 QSO sintetici distribuiti sugli ultimi sei mesi:

```bash
.venv/bin/python scripts/seed_demo_qsos.py
```

Lo script crea prima un backup e marca ogni riga con `[DEMO-6M]`. Non duplica i dati già presenti; l'opzione `--replace-demo` serve a rigenerarli.

## Articolo

[I Built an AI Ham Radio Assistant That Runs Entirely Offline](https://medium.com/@andrea.maccafeo/i-built-an-ai-ham-radio-assistant-that-runs-entirely-offline-because-your-qso-log-is-none-of-992cac4230cf) — Medium
