# Requisiti di sistema

Non esiste ancora un requisito minimo certificato per il Consigliere di Stazione. Per dichiararlo servirebbero prove ripetibili su più sistemi operativi e computer con quantità diverse di RAM. Questa pagina separa quindi i dati verificati dalle indicazioni pratiche.

## Dati verificati

| Voce | Dato verificato |
| --- | --- |
| Eseguibile Windows | Formato `PE32+ x86-64`, circa 21,4 MB nella build attuale |
| Build e test locali | Python 3.12 |
| Modello predefinito | `qwen3.5:4b`, quantizzazione `Q4_K_M`, download Ollama da 3,4 GB |
| Modello caricato | 3,1 GB indicati da `ollama ps`, con contesto 4096 |
| Modello embedding RAG | `nomic-embed-text`, 323 MB indicati da `ollama ps` |
| Indice tecnico attuale | Circa 20 MB in ChromaDB |

Durante una prova locale con Qwen e il modello embedding caricati insieme, i due processi Ollama hanno occupato circa 4,25 GB di VRAM. Questo valore non comprende il sistema operativo, il browser, il server RAG e il resto dell'applicazione. Su un computer senza GPU, una parte maggiore del carico ricade sulla RAM di sistema.

La storia del progetto riporta prove su PC `x86_64`, Raspberry Pi 4 con almeno 4 GB e Raspberry Pi 5. Non sono invece stati documentati test sufficienti per fissare Windows 10, Python 3.10 o una specifica quantità di memoria come minimo garantito.

## Indicazioni pratiche

### Solo applicazione

Questa modalità comprende registro QSO, storico, esportazioni, backup, NOAA, POTA e consigli calcolati dalle regole.

- **4 GB di RAM** sono una base prudente perché corrispondono alla configurazione Raspberry Pi più piccola già riportata dal progetto; non sono un minimo misurato.
- **8 GB di RAM** lasciano più margine al sistema operativo e al browser, ma l'applicazione da sola non è stata sottoposta a un benchmark che dimostri di richiederli.
- L'eseguibile fornito è per Windows a 64 bit. La versione minima di Windows non è ancora stata certificata.
- Su Linux, Python 3.12 è la versione verificata. Altre versioni possono funzionare, ma non sono indicate qui come supportate finché non vengono provate dalla suite.

Il registro funziona senza Internet. La connessione serve soltanto per aggiornare NOAA, leggere gli spot POTA o raggiungere eventuali servizi AI remoti.

### Applicazione, Qwen e RAG sullo stesso computer

Il vecchio valore di 4 GB proveniva dal precedente modello `amaccafeo/swlbot` da 1,9 GB. Non descrive correttamente la configurazione attuale con `qwen3.5:4b`.

- **8 GB di RAM** sono una soglia pratica da cui partire, non un minimo garantito. La memoria effettivamente libera, il sistema operativo, il contesto del modello e l'uso di CPU o GPU possono fare la differenza.
- **16 GB di RAM** sono la configurazione consigliata per lasciare margine quando Consigliere, Ollama e swlbot RAG lavorano sulla stessa macchina.
- Per il disco considera almeno il modello da 3,4 GB, il modello embedding, l'indice ChromaDB e lo spazio necessario a Ollama per aggiornamenti o modelli alternativi.
- La GPU non è obbligatoria. Ollama può usare la CPU, con tempi di risposta generalmente più lunghi.

Queste sono raccomandazioni operative ricavate dai componenti attuali, non una certificazione delle prestazioni su ogni computer.

### AI su un altro computer

Se Qwen e swlbot RAG girano su una macchina separata, il computer che ospita il Consigliere deve sostenere soltanto l'applicazione base. Il carico del modello resta sul server AI. Prima di configurare un indirizzo remoto, leggi la sezione privacy del README: il riepilogo operativo viene inviato all'endpoint impostato.

## Installare i componenti

### Consigliere di Stazione

Su Windows scarica l'eseguibile dalla pagina [GitHub Releases](https://github.com/111blackeagle111/consigliere-di-stazione/releases) e segui la [guida rapida](GUIDA_WINDOWS.txt). Python non serve.

Su Linux e Raspberry Pi usa i comandi riportati nella sezione Installazione del [README](../README.md#linux-e-raspberry-pi). Quella procedura crea un ambiente virtuale e installa le dipendenze elencate dal progetto.

### Ollama e Qwen

Ollama cambia nel tempo requisiti, installer e supporto hardware. Per questo il progetto non conserva una copia delle sue istruzioni:

1. apri la pagina ufficiale [Download Ollama](https://ollama.com/download);
2. scegli il tuo sistema operativo e segui la relativa guida ufficiale per [Windows](https://docs.ollama.com/windows) o [Linux](https://docs.ollama.com/linux);
3. al termine, apri un nuovo terminale e verifica l'installazione con:

   ```text
   ollama -v
   ```

Su Windows l'installer ufficiale avvia normalmente Ollama in background. Su Linux, modalità di avvio e servizio sono descritte nella guida ufficiale. Quando Ollama è attivo, la sua API locale risponde sulla porta `11434`, che è anche l'indirizzo predefinito usato dal Consigliere.

Scarica quindi il modello configurato dall'app:

```text
ollama pull qwen3.5:4b
```

La scheda ufficiale del modello è [qwen3.5:4b su Ollama](https://ollama.com/library/qwen3.5:4b). Puoi controllare che il download sia presente con `ollama ls` e provarlo con `ollama run qwen3.5:4b`; per uscire dalla chat digita `/bye`.

A questo punto il fallback Qwen diretto del Consigliere è pronto. Non serve installare il RAG per usarlo: se swlbot RAG non è disponibile, l'interfaccia mostra **CONSIGLIO QWEN SENZA RAG**.

### swlbot RAG, facoltativo e in sviluppo

swlbot RAG aggiunge il corpus tecnico, ma è un componente separato e non è incluso nell'eseguibile Windows. Se possiedi già la sua directory di progetto, prepara il modello embedding:

```text
ollama pull nomic-embed-text
```

La scheda ufficiale è [nomic-embed-text su Ollama](https://ollama.com/library/nomic-embed-text). Installa poi le dipendenze indicate dal file `requirements.txt` di swlbot RAG e avvia `server.py` sulla porta `8081`. Finché il componente resta un lavoro in corso, la sua guida specifica ha la precedenza su queste indicazioni generali.

Per controllare i due servizi:

- `http://127.0.0.1:11434` è l'API locale di Ollama;
- `http://127.0.0.1:8081/health` restituisce lo stato di swlbot RAG;
- `http://127.0.0.1:8080` apre il Consigliere di Stazione.

Se Ollama modifica la procedura di installazione, fa sempre fede la sua [documentazione ufficiale](https://docs.ollama.com/quickstart).

## Cosa manca per dichiarare un minimo ufficiale

Prima della release definitiva sarebbe utile provare almeno:

- avvio, registro ed esportazioni su Windows 10 e Windows 11 a 64 bit;
- suite completa su più versioni Python;
- applicazione base su una macchina con 2 GB e una con 4 GB di RAM;
- Qwen diretto e RAG completo su sistemi con 8 GB e 16 GB;
- Raspberry Pi 4 e 5 con tempi di risposta e consumo memoria annotati.

Fino a quel collaudo, questa pagina evita volutamente di presentare come minimo un numero che non è stato misurato.
