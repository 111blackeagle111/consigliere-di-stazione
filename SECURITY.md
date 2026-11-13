# Sicurezza

Consigliere di Stazione è un’app personale senza autenticazione. Il server ascolta su `127.0.0.1` per impostazione predefinita. L’ascolto sulla LAN è una scelta esplicita per reti fidate. L’esposizione diretta a Internet è fuori dal modello di distribuzione supportato.

- Proteggi il registro QSO, il nominativo, il locator, gli export e il backup SQLite. Verifica che una pagina di un altro sito non possa leggere dati privati o provocare scritture indesiderate. Considera gli effetti effettivi degli endpoint, inclusi i `GET` che possono salvare record tecnici.
- Tratta come non attendibili i dati dei moduli, degli spot POTA e delle risposte AI. Verifica che vengano visualizzati come testo e che i campi esportati in CSV non possano essere interpretati come formule.
- Le richieste a NOAA e POTA devono essere prive di dati del registro. Ai servizi RAG e Ollama configurati possono arrivare il locator, le statistiche aggregate necessarie al consiglio e i dati NOAA/POTA. Escludi note e singoli record QSO dalle richieste a questi servizi. Export e backup avviati dall’utente fanno parte delle funzioni previste.
- Per modifiche a GitHub Actions e alle release, verifica che il codice di una PR non possa accedere a credenziali di pubblicazione o sostituire l’eseguibile distribuito.

Per ogni segnalazione descrivi il percorso raggiungibile, i prerequisiti e l’impatto. Specifica se lo scenario riguarda localhost, l’ascolto esplicito sulla LAN o la pipeline di rilascio.
