# Revisione del codice

- Per modifiche al database o al launcher, verifica che l’aggiornamento ritrovi il registro SQLite esistente, incluso quello accanto all’eseguibile Windows. Controlla che storico, statistiche ed export continuino a separare i QSO manuali dai record tecnici di NOAA e degli alert.
- Per modifiche a NOAA, POTA o ai consigli, verifica che l’interfaccia distingua i dati appena acquisiti, la cache ancora valida e l’ultimo valore valido recuperato dopo un errore. Verifica che il consiglio resti disponibile tramite le regole locali quando i servizi AI sono indisponibili o le loro risposte vengono scartate. Il controllo degli alert si basa sui dati NOAA/POTA senza consultare lo storico QSO.
- Per modifiche al packaging o ai workflow, verifica che l’eseguibile Windows includa le risorse necessarie e che i test restino attivi sulle PR e prima delle release.

Per ogni rilievo indica la condizione che innesca il problema, il comportamento atteso e l’impatto osservabile.
