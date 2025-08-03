Dieses Repository enthält den vollständigen Quellcode und die Dokumentationsstrukturen für den im Rahmen meiner Masterarbeit entwickelten Prototypen zur semantischen Dokumentenrecherche im kommunalen Finanzbereich. 
Ziel ist es, einen Beitrag zur Wissenssicherung und Effizienzsteigerung im Bereich Finanzen und Steuern kleiner Kommunalverwaltungen durch moderne KI-Methoden zu leisten.

Das System implementiert einen Retrieval-Augmented Generation (RAG)-Ansatz, bei dem externe Dokumente (wie Satzungen, Richtlinien, Tabellen) automatisiert eingelesen, semantisch indexiert und über ein Large Language Model (LLM) für die Beantwortung fachlicher Anfragen genutzt werden. 
Im Sinne der wissenschaftlichen Nachvollziehbarkeit erfolgt die Bereitstellung aller Codes und Daten über dieses öffentlich zugängliche Repository.

Projektstruktur:
Das Repository ist klar strukturiert und umfasst folgende zentrale Elemente:

main.py bildet den Einstiegspunkt und enthält die FastAPI-Logik zur Bereitstellung der API-Endpunkte für das System.

Im Ordner utils/ befinden sich sämtliche Hilfsmodule für die Dokumentenextraktion (extractors.py), das Laden verschiedener Dateiformate (loader.py), die semantische Suche (retriever.py) und die Verwaltung der Vektordatenbank (vectorstore.py).

Der Ordner data/ enthält exemplarische Testdokumente (PDF, DOCX, XLSX), mit denen das System überprüft werden kann.

Im Verzeichnis vectorstore/ werden die persistenten Vektorindizes abgelegt, die von FAISS erzeugt werden.

Die Datei requirements.txt listet sämtliche benötigten Python-Abhängigkeiten auf.

Diese README erklärt die Funktion und Nutzung des Prototyps.

Systemübersicht und Funktionalität:
Der Prototyp wurde auf Basis umfangreicher Literaturrecherche, der Auswertung bestehender Implementierungsbeispiele und im Abgleich mit aktuellen wissenschaftlichen Veröffentlichungen konzipiert. Eine zentrale technische Orientierung bot das Tutorial von Anand (2023), das exemplarisch die Integration von FastAPI, FAISS, HuggingFace-Embeddings und einem LLM-API-Backend zeigt. Die eigentliche Umsetzung wurde um zahlreiche projektspezifische Erweiterungen ergänzt, darunter eine mehrstufige Extraktions- und Chunking-Logik, ein hybrider Retriever, sowie eine an den kommunalen Anwendungskontext adaptierte Systemkonfiguration. Teile des Entwicklungsprozesses wurden zudem durch GPT-4o-gestützte Hilfestellungen unterstützt.

Die Anwendung läuft als modularer FastAPI-Server, der Anfragen über definierte REST-Endpunkte entgegennimmt. Das Frontend kann durch eine einfache, barrierearme HTML-Oberfläche realisiert werden, um den Zugang für Endanwender:innen ohne technische Vorkenntnisse sicherzustellen. Die technische Infrastruktur basiert auf einer Microsoft Azure VM (Standard B2ms), um die praktische Übertragbarkeit in kommunale IT-Landschaften zu gewährleisten.

Technische Kernpunkte:

Dokumentenverarbeitung:
Die Extraktion erfolgt zweistufig: Zunächst wird mit PyPDF2 der eingebettete Text extrahiert. Ist dies nicht möglich (z. B. bei gescannten Dokumenten), greift automatisiert Tesseract-OCR. Nach der Extraktion werden die Texte entlang typischer Verwaltungsgliederungen (z. B. Paragraphen, Abschnitte) segmentiert und mit Metadaten angereichert.

Dateiformate:
Das System unterstützt PDF, DOCX, XLSX, XLSM und XLS, sodass ein Großteil kommunaler Verwaltungsvorlagen verarbeitet werden kann.

Semantischer Index:
Die semantische Suche basiert auf FAISS, das für Effizienz und Skalierbarkeit auch bei größeren Korpora sorgt. Die Embeddings werden mit sentence-transformers/all-MiniLM-L6-v2 erzeugt, da dieses Modell eine sehr gute Balance zwischen Geschwindigkeit und inhaltlicher Präzision bei deutschen Verwaltungstexten bietet.

Hybrid Retrieval:
Die eigentliche Suche kombiniert dichte (semantische) und spärliche (BM25-Keyword-basierte) Methoden. Die Gewichtung (0,3 dense / 0,7 sparse) wurde experimentell bestimmt und stellt die optimale Balance zwischen semantischer Relevanz und exakter Begriffstreue sicher.

Antwortgenerierung:
Für die Generierung der Antworten wird das LLM meta-llama/llama-4-scout-17b-16e-instruct via Groq-API verwendet. Die Modellparameter (max_tokens=1024, temperature=0.0, top_p=0.9) sind so gewählt, dass die Antworten konsistent, faktenbasiert und in einer angemessenen Länge bereitgestellt werden. Der Prompt verpflichtet das Modell zur strukturierten, referenzbasierten Wiedergabe der gefundenen Inhalte.

Installation und Anwendung:

Das Repository kann mit git clone geklont und in einer virtuellen Umgebung mit den in requirements.txt gelisteten Paketen installiert werden.

Der Serverstart erfolgt über den Befehl uvicorn main:app --reload.

Dokumente werden über die APIs geladen, verarbeitet und können dann für Anfragen genutzt werden. Die wichtigsten Endpunkte sind /test-dokumente, /build und /ask.

Wissenschaftlicher Kontext und Nachvollziehbarkeit:
Dieses Repository dient als technische Grundlage zur wissenschaftlichen Evaluation von KI-basierten RAG-Systemen in der kommunalen Verwaltung. 
Die klare Trennung von Code, Daten und Dokumentation, sowie die konsequente Kommentierung der Module ermöglichen eine vollständige Nachvollziehbarkeit und eine einfache Erweiterung des Systems im Rahmen weiterführender Forschungs- oder Praxisprojekte.

Literaturhinweis & Anfertigungshinweis:
Die grundlegende technische Orientierung erfolgte gemäß:
Anand, S. (2023). AI-Powered Q&A Chatbot with LangChain & FastAPI. Medium. https://medium.com/@shivanshuk33/ai-powered-q-a-chatbot-with-langchain-fastapi-c07e1796efad

Der Inhalt dieser Readme wurde basierend auf meinen Angaben und meinem Wissen von ChatGPT 4o generiert. - https://chatgpt.com/
