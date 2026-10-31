"""Etiquetas débiles por reglas multilingües sobre el informe."""
import os
import re
import unicodedata

import pandas as pd

from config import ETIQUETAS as OUT, LABELS, TRAIN_CSV

HIGH, PLAIN, LOW, NOT_MENTIONED, NEG = .9, .7, .35, .1, .03


def norm(s):
    s = unicodedata.normalize("NFKD", str(s).replace("İ", "i").replace("I", "i").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.replace("µ", "μ").replace("ς", "σ").replace("∆", "δ").replace("Ω", "ω")


def rx(words, boundary=True):
    """Prefijos (raíces) normalizados → regex que exige inicio de palabra y deja libre la terminación."""
    alts = "|".join(sorted({re.escape(norm(w)) for w in words}, key=len, reverse=True))
    return re.compile((r"(?<![\w])(?:" if boundary else "(?:") + alts + ")")


# ---------------- vocabulario ----------------
# Patología "rotura / desgarro"
TEAR = rx(["tear", "torn", "rupture", "ruptur", "disrupt", "discontinu", "bucket", "flap", "avulsi",
           "rotura", "roto", "desgarr", "ruptura", "asa de cubo", "lesion", "lesión",
           "riss", "ruptur", "gerissen", "einriss", "korbhenkel", "lappen",
           "yırtı", "yirti", "rüptür", "ruptur", "kopuk", "kopma", "kova sap", "devamsızlık", "devamsizlik",
           "ruptur", "puknu", "prekid", "lezij", "rascjep", "ručka", "rucka",
           "ρηξ", "ρηγμ", "διακοπ", "αποκοπ",
           "руптур", "разкъс", "скъсан", "скъсване", "лезия", "прекъсван", "дръжка на кофа",
           "scheur", "ruptuur", "ruptur", "afscheur", "emmerhengsel", "letsel",
           "dechirure", "déchirure", "rupture", "fissure meniscale", "anse de seau", "lesion"])
# Rotura de menisco: además "grado 3", "llega a la superficie"
MENISCUS_EXTRA = rx(["grade 3", "grade iii", "grado 3", "grado iii", "grad 3", "grad iii", "grade 3 ", "stupanj 3",
                     "iii. stup", "3. stup", "βαθμου iii", "βαθμου 3", "iii степен", "3 степен", "graad 3", "graad iii",
                     "reaching the articular surface", "extending to the articular surface", "surface tear",
                     "contacta con la superficie", "alcanza la superficie", "eklem yüzeyine uzanan",
                     "eklem yuzeyine uzanan", "doseže zglobnu", "dopire do zglobne",
                     "φθανει στην αρθρικη", "достига ставн", "atteint la surface", "bereikt het gewrichtsoppervlak",
                     "truncat", "macerat", "extrusi", "displaced fragment", "fragmento desplazado", "substanzgemindert",
                     "substanzverlust", "teilresekt", "resekt", "resect", "meniscectom", "menisectom", "meniskektom",
                     "rezeksiyon", "parsiyel menisektomi", "resekcij", "μηνισκεκτομ", "резекция", "менискектом",
                     "meniscectomie", "méniscectomie", "complex", "komplex", "kompleks", "radial", "radiär", "radiar",
                     "horizontal", "horizontaal", "vertical", "vertikal", "longitudinal", "oblique", "schräg", "schrag"])
# Estructuras
ACL = rx(["anterior cruciate", "acl", "lca", "ligamento cruzado anterior", "cruzado anterior", "cruciate anterior",
          "vorderes kreuzband", "vorderen kreuzband", "vordere kreuzband", "vkb",
          "ön çapraz", "on capraz", "ön carpraz", "ack", "ön çaprazbağ", "anterior çapraz", "anterior capraz",
          "prednji križni", "prednjeg križnog", "prednji kriz", "prednjeg kriz", "lca", "prednja ukrsten", "prednjeg ukrst",
          "προσθιο χιαστο", "προσθιου χιαστου", "προσθιοσ χιαστοσ", "πχσ",
          "предна кръстна", "предната кръстна", "пкв", "предна кръстосана",
          "voorste kruisband", "vkb",
          "croise anterieur", "croisé antérieur", "lca"])
MCL = rx(["medial collateral", "mcl", "tibial collateral", "ligamento colateral medial", "ligamento colateral interno",
          "colateral medial", "colateral interno", "lcm", "lli",
          "mediales kollateralband", "mediale kollateralband", "medialen kollateralband", "innenband", "mediales seitenband",
          "medialen seitenband",
          "medial kollateral", "medial kolateral", "iç yan bağ", "ic yan bag", "mkl",
          "medijalni kolateralni", "medijalnog kolateralnog", "medijalni kolateraln", "medijalnog kolateraln", "unutarnji kolateral",
          "medijalnu kolateraln", "medijalna kolateraln", "medijalne kolateraln", "medijalni kolateral", "medijalnog kolateral",
          "εσω πλαγιο", "εσω πλαγιου", "εσωτερικο πλαγιο", "εσωτερικου πλαγιου",
          "медиална колатерална", "медиалната колатерална", "медиален колатерален", "вътрешна колатерална",
          "mediale collaterale", "mediaal collateraal", "mediale collateraal", "binnenband",
          "collateral interne", "collatéral interne", "collateral medial", "lli", "ligament lateral interne"])
MENISCUS = rx(["menisc", "menisk", "menisküs", "meniskus", "menisci", "ménisque", "menisque", "μηνισκ", "менискус", "мениск",
               "meniscus", "meniscal", "meniskopati"], boundary=False)
MEDIAL = rx(["innenmenisk", "medial", "internal", "interno", "interna", "innen", "medial", "iç ", "ic ", "medyal", "medijaln", "unutarnj",
             "εσω", "εσωτερικ", "медиал", "вътреш", "mediale", "mediaal", "binnen", "interne", "médial", "medial"])
LATERAL = rx(["außenmenisk", "aussenmenisk", "lateral", "external", "externo", "externa", "außen", "aussen", "lateral", "dış ", "dis ", "lateraln", "vanjsk",
              "εξω", "εξωτερικ", "латерал", "външ", "laterale", "lateraal", "buiten", "externe", "latéral"])
# Cartílago / artrosis
OA_PATH = rx(["osteoarthr", "arthrosis", "arthrotic", "chondromalac", "chondral loss", "cartilage loss", "full thickness",
              "full-thickness", "denuded", "chondral defect", "cartilage defect", "chondropath",
              "artrosis", "artrósic", "artrosic", "condromalac", "condropat", "perdida de cartilago", "pérdida de cartílago",
              "defecto condral", "espesor completo",
              "arthrose", "chondromalaz", "chondropath", "knorpelschaden", "knorpelverlust", "knorpeldefekt", "knorpelglatze",
              "vollschicht",
              "artroz", "gonartroz", "kondromalaz", "kıkırdak kayb", "kikirdak kayb", "kıkırdak defekt", "kondropati",
              "tam kat", "kıkırdak incel", "kikirdak incel",
              "artroz", "gonartroz", "hondromalac", "hondropat", "defekt hrskav", "gubitak hrskav", "hondraln",
              "οστεοαρθρ", "αρθροπαθ", "αρθρισ", "χονδρομαλακ", "χονδροπαθ", "απωλεια χονδρ", "λεπτυνση χονδρ",
              "χονδρικ ελλειμ", "πληρουσ παχουσ",
              "артроз", "гонартроз", "хондромалац", "хондропат", "хрущялен дефект", "загуба на хрущял", "изтъняване на хрущял",
              "artrose", "chondromalac", "chondropath", "kraakbeenlijden", "kraakbeendefect", "kraakbeenverlies", "volle dikte",
              "arthrose", "chondropath", "chondromalac", "perte de cartilage", "perte cartilag", "amincissement", "defect chondral",
              "défect chondral", "pleine epaisseur", "pleine épaisseur",
              "thinning", "thin articular cartilage", "thinned cartilage", "chondral ulcer", "cartilage ulcer", "erosion", "eroded",
              "osteophyt", "spur", "fissur", "denud", "bone on bone", "joint space narrowing",
              "adelgazamiento", "ulcera condral", "úlcera condral", "ulceras condrales", "úlceras condrales", "erosion", "osteofit",
              "pinzamiento", "denudad", "knorpelausdünnung", "knorpelausdunnung", "ausdünnung", "ausdunnung", "knorpelulkus",
              "osteophyt", "gelenkspaltverschmäler", "gelenkspaltverschmaler", "incelme", "ülser", "ulser", "erozyon", "osteofit",
              "eklem aralığında daral", "eklem araliginda daral", "denudacij", "eroz", "ulceracij", "osteofit", "suženje zglobn",
              "suzenje zglobn", "λεπτυνσ", "διαβρωσ", "εξελκωσ", "οστεοφυτ", "στενωση του μεσαρθριου", "изтъняван", "ерози",
              "остеофит", "стесняване на ставната", "verdunning", "ulceratie", "kraakbeen letsel", "kraakbeenletsel", "osteofyt",
              "gewrichtsspleetversmalling", "érosion", "ulcération", "ulceration", "ostéophyt", "osteophyt", "pincement",
              "oa ", "oa,", "artrosis femoro", "ofa "])
PF = rx(["patell", "trochle", "femoropatell", "femoro-patell", "patelofemoral", "patellofemoral", "retropatell", "rotul",
         "kneecap", "kniescheibe", "diz kapa", "patela", "pately", "patelar", "επιγονατ", "τροχιλ", "пател", "трохле",
         "knieschijf", "rotule", "trochlée", "fémoro-patell", "femoro-patell", "trohle", "troklea", "facet", "faset",
         "faceta", "facette", "pf ", "pf-", "femoropatelar", "femoropatellaar", "patellofemoraal"])
COMP_MED = rx(["medial compartment", "medial femorotibial", "medial femoral condyle", "medial tibial plateau",
               "medial femoral", "medial tibial", "compartimento medial", "compartimento interno", "condilo femoral medial",
               "cóndilo femoral medial", "condilo femoral interno", "platillo tibial medial", "platillo tibial interno",
               "femorotibial medial", "femorotibial interna", "medialen femur", "medialer femur", "mediales kompartiment",
               "medialen kompartiment", "medialen tibia", "medialer tibia", "innenkompartiment",
               "medial femoral kondil", "medial tibial plato", "medial kompartman", "medial femorotibial", "iç kompartman",
               "medijalni kondil", "medijalnog kondila", "medijalni odjeljak", "medijalnog odjeljka", "medijalni femorotibijal",
               "medijalnog femorotibijal", "medijalnog platoa", "medijalni plato",
               "εσω μηριαι", "εσω κονδυλ", "εσω διαμερισμα", "εσω κνημιαι", "εσω μηροκνημιαι",
               "медиален кондил", "медиалния кондил", "медиален компартимент", "медиалния компартимент", "медиално плато",
               "медиалното плато", "медиална феморотибиална",
               "mediale compartiment", "mediaal compartiment", "mediale femurcondyl", "mediale tibiaplateau", "mediaal femorotibiaal",
               "compartiment interne", "compartiment médial", "condyle interne", "condyle médial", "plateau tibial interne",
               "plateau tibial médial", "fémoro-tibial interne", "femoro-tibial interne", "femorotibial interne"])
COMP_LAT = rx([w.replace("medial", "lateral").replace("interno", "externo").replace("interna", "externa")
               .replace("innen", "außen").replace("iç ", "dış ").replace("medijaln", "lateraln").replace("εσω", "εξω")
               .replace("медиал", "латерал").replace("mediale", "laterale").replace("mediaal", "lateraal")
               .replace("interne", "externe").replace("médial", "latéral") for w in COMP_MED.pattern.split("|")[:0]] +
              ["lateral compartment", "lateral femorotibial", "lateral femoral condyle", "lateral tibial plateau",
               "lateral femoral", "lateral tibial", "compartimento lateral", "compartimento externo", "condilo femoral lateral",
               "cóndilo femoral lateral", "condilo femoral externo", "platillo tibial lateral", "platillo tibial externo",
               "femorotibial lateral", "femorotibial externa", "lateralen femur", "lateraler femur", "laterales kompartiment",
               "lateralen kompartiment", "lateralen tibia", "lateraler tibia", "außenkompartiment", "aussenkompartiment",
               "lateral femoral kondil", "lateral tibial plato", "lateral kompartman", "dış kompartman",
               "lateralni kondil", "lateralnog kondila", "lateralni odjeljak", "lateralnog odjeljka", "lateralni femorotibijal",
               "lateralnog femorotibijal", "lateralnog platoa", "lateralni plato",
               "εξω μηριαι", "εξω κονδυλ", "εξω διαμερισμα", "εξω κνημιαι", "εξω μηροκνημιαι",
               "латерален кондил", "латералния кондил", "латерален компартимент", "латералния компартимент", "латерално плато",
               "латералното плато", "латерална феморотибиална",
               "laterale compartiment", "lateraal compartiment", "laterale femurcondyl", "laterale tibiaplateau",
               "lateraal femorotibiaal",
               "compartiment externe", "compartiment latéral", "condyle externe", "condyle latéral", "plateau tibial externe",
               "plateau tibial latéral", "fémoro-tibial externe", "femoro-tibial externe", "femorotibial externe"])
GENERAL_OA = rx(["gonarthr", "tricompartment", "tricompartiment", "three compartments", "los tres compartimentos",
                 "gonartroz", "gonartroz", "trikompartiment", "τρια διαμερισματ", "гонартроз", "drie compartiment",
                 "trois compartiments", "all three compartments", "all compartments", "todos los compartimentos"])
# Entidades propias
EFFUSION = rx(["effusion", "hydrarthros", "joint fluid", "intra-articular fluid", "intraarticular fluid", "fluid in the joint",
               "derrame", "liquido articular", "líquido articular", "liquido intraarticular", "líquido intraarticular",
               "hidrartros", "erguss", "gelenkerguss", "flüssigkeit", "flussigkeit",
               "efüzyon", "efuzyon", "sıvı", "sivi", "mayi",
               "izljev", "tekucin", "tekućin", "efuzij", "hidrops",
               "συλλογη", "υγρου", "υδαρθρ",
               "излив", "течност", "хидроартроз", "хидропс",
               "hydrops", "gewrichtsvocht", "vocht", "effusie",
               "epanchement", "épanchement", "liquide articulaire", "hydarthrose",
               "hemarthros", "haemarthros", "lipohemarthros", "hämarthros", "hamarthros", "liphämarthros", "liphamarthros",
               "hemartros", "hemartroz", "lipohemartroz", "αιμαρθρ", "λιποαιμαρθρ", "хемартроз", "хемартрос", "hemartrose",
               "hémarthrose", "lipohémarthrose"], boundary=False)
HEMARTHROS = rx(["hemarthros", "haemarthros", "hämarthros", "hamarthros", "hemartros", "hemartroz", "αιμαρθρ", "хемартроз",
                 "hémarthrose", "hemartrose"], boundary=False)
SYNOVITIS = rx(["synovitis", "synovial thickening", "synovial hypertrophy", "synovial proliferation", "thickened synovium",
                "sinovitis", "engrosamiento sinovial", "hipertrofia sinovial", "synovialitis", "synovitis",
                "synoviale verdickung", "synovialverdickung", "sinovit", "sinoviyal kalınlaş", "sinoviyal kalinlas",
                "sinovijalno zadeblj", "zadebljanje sinovij", "sinovitis", "συνοβιτ", "υμενιτ", "παχυνση του αρθρικου υμενα",
                "синовит", "задебеляване на синовия", "synovitis", "synoviale verdikking", "synovite",
                "épaississement synovial", "epaississement synovial"])
BAKER = rx(["baker", "popliteal cyst", "quiste popliteo", "quiste poplíteo", "poplitealzyste", "kniekehlenzyste", "popliteal kist",
            "bakerova", "poplitealna cista", "μπεικερ", "ιγνυακη κυστη", "κυστη baker", "бейкер", "поплитеална киста",
            "kyste poplite", "kyste poplité", "popliteacyste", "kniekuilcyste"])
CONTUSION = rx(["bone-bruise", "bonebruise", "koštan edem", "koštanog edem", "koštani edem", "kostan edem", "kostanog edem",
                "kostani edem", "postkontuzij", "ödemsignal", "odemsignal", "bone marrow signal", "bone bruise", "bone contusion", "contusion", "marrow edema", "marrow oedema", "bone edema", "bone oedema",
                "contusion osea", "contusión ósea", "contusion", "edema oseo", "edema óseo", "edema de medula", "edema medular",
                "knochenmarködem", "knochenmarkodem", "knochenmarksödem", "knochenmarksodem", "knochenprellung", "kontusion",
                "bone bruise", "kemik iliği ödem", "kemik iligi odem", "kemik ödem", "kemik iliginde ödem", "kemik iliğinde ödem",
                "kontüzyon", "kontuzyon",
                "kontuzij", "edem koštane srži", "edem kostane srzi", "edem kosti", "edem spongioz", "nagnjeće", "nagnjece",
                "θλαση", "οιδημα του οστικου μυελου", "οιδημα οστικου μυελου", "οστικο οιδημα", "οστικη θλαση", "οιδημα μυελου",
                "контузи", "оток на костния мозък", "костно-мозъчен оток", "костномозъчен оток", "костен оток", "едем на костния",
                "костномозъчен едем", "костен едем",
                "botkneuzing", "kneuzing", "botoedeem", "beenmergoedeem", "contusie",
                "contusion osseuse", "œdème osseux", "oedeme osseux", "edeme osseux", "œdème médullaire", "oedeme medullaire"])
FRACTURE = rx(["fractur", "fracture line", "fractura", "fraktur", "frakt", "kırık", "kirik", "fraktür", "fraktur",
               "prijelom", "prelom", "fraktur", "καταγμ", "фрактур", "счупван", "fractuur", "breuk", "fracture"])

# Calificadores
NEGATION_BEFORE = rx(["no ", "not ", "without", "negative for", "absence of", "absent", "no evidence", "ruled out", "free of",
                      "sin ", "no se ", "ausencia", "descarta", "kein", "keine", "ohne", "nicht ", "kein hinweis",
                      "bez ", "nema ", "nije ", "ne nalazi", "ninguno", "ninguna", "ne prikazuje", "χωρισ", "δεν ", "ουδεν", "ουτε",
                      "без ", "няма ", "не се ", "липсва", "geen ", "zonder", "niet ", "pas d", "pas de", "sans ", "aucun", "ni "])
NEGATION_AFTER = rx(["yok", "saptanmad", "izlenmed", "saptanmamıs", "saptanmamis", "izlenmemis", "izlenmemıs",
                     "gorulmemis", "görülmemis", "rastlanmamıs", "rastlanmamis", "secilmemis", "seçilmemis", "olmayan",
                     "izlenmemektedir", "saptanmamaktadır", "saptanmamaktadir", "görülmed", "gorulmed", "rastlanmad", "seçilmed", "secilmed",
                     "izlenmemekte", "saptanmamakta", "görülmemekte", "gorulmemekte", "mevcut değil", "mevcut degil",
                     "bulunmamakta", "ayırt edilmed", "ayirt edilmed",
                     "not seen", "not identified", "not detected", "not detect", "is not ", "not present", "absent",
                     "nicht nachweisbar", "nicht abgrenzbar", "ausgeschlossen", "nije prisut", "nije vidljiv", "ne vidi",
                     "δεν αναγνωριζ", "δεν απεικονιζ", "не се визуализира", "не се установява", "не се открива",
                     "niet aantoonbaar", "afwezig", "n'est pas", "absente", "non visible", "none", "geen", "aucun", "keine", "nein",
                     "nema", "няма", "ninguno", "negative", "negatief", "négatif", "negativ", "ausente", "absent"])
NORMAL = rx(["normal", "intact", "unremarkable", "preserved", "integr", "íntegr", "conservad", "intakt", "regelrecht",
             "unauffällig", "unauffallig", "normaldir", "olağan", "olagan", "doğal", "dogal", "korunmuş", "korunmus", "intakt",
             "uredn", "primjeren", "očuvan", "ocuvan", "intakt", "φυσιολογικ", "ακεραι", "нормал", "запазен", "б.о.",
             "intact", "normaal", "intact", "sans particularit", "normal", "unauffällig", "physiolog", "fisiolog", "fizyolojik",
             "fiziolo", "φυσιολογ", "физиолог", "fysiolog", "physiologique"])
LOW_Q = rx(["tiny", "sitn", "partial", "low grade", "low-grade", "mild", "small", "minimal", "minor", "trace", "slight", "sprain", "strain",
          "grade 1", "grade i ", "grade i,", "grade 2", "grade ii", "grade i-ii", "grade 1-2", "degenerat", "mucoid",
          "physiolog", "incipient", "early", "beginning", "intrasubstance", "intrasubstance", "fraying", "superficial",
          "chronic", "old", "healed", "sclerotic",
          "parcial", "bajo grado", "leve", "pequeñ", "pequen", "minim", "escaso", "discret", "esguince", "distension",
          "grado 1", "grado i ", "grado 2", "grado ii", "degenerativ", "mucoide", "fisiologic", "incipiente", "inicial",
          "cronic", "antigu", "consolidad",
          "partiell", "teilriss", "gering", "klein", "minimal", "leicht", "diskret", "zerrung", "distorsion", "grad 1", "grad i ",
          "grad 2", "grad ii", "degenerativ", "mukoid", "physiologisch", "beginnend", "chronisch", "alt", "verheilt",
          "kısmi", "kismi", "parsiyel", "hafif", "az miktar", "minimal", "küçük", "kucuk", "sınırlı", "sinirli", "grade 1",
          "grade 2", "grade ii", "grade i ", "evre 1", "evre 2", "dejenerasyon", "dejeneratif", "erken", "kronik", "eski",
          "iyileşmiş", "iyilesmis", "fizyolojik", "burkulma",
          "parcijaln", "djelomičn", "djelomicn", "blag", "mal", "minimal", "malo ", "male ", "početn", "pocetn",
          "i. stup", "ii. stup", "i/ii", "i°", "ii°", "1. stup", "2. stup", "degenerativ", "mukoidn", "distenzij",
          "kroničn", "kronicn", "star", "fiziološk", "fizioloski", "istegnut",
          "μερικ", "υφολικ", "ηπια", "ηπιο", "μικρη", "μικρο", "μικρου", "ελαχιστ", "βαθμου i ", "βαθμου ii", "βαθμου 1",
          "βαθμου 2", "εκφυλιστ", "βλεννωδ", "αρχομεν", "παλαι", "χρονι", "διαστρεμμα", "φυσιολογικ ποσοτ",
          "парциал", "частич", "лек", "малък", "малко", "минимал", "незначител", "i степен", "ii степен", "1 степен", "2 степен",
          "дегенератив", "мукоид", "начал", "хронич", "стар", "физиологич", "разтягане",
          "partieel", "partiële", "partiele", "gering", "klein", "minimaal", "licht", "lichte", "graad 1", "graad i ", "graad 2",
          "graad ii", "degeneratie", "degeneratief", "mucoid", "beginnend", "fysiologisch", "chronisch", "oud", "verstuiking",
          "partiel", "partielle", "léger", "leger", "légère", "legere", "petit", "minime", "discret", "entorse", "grade 1",
          "grade 2", "stade 1", "stade 2", "dégénérat", "degenerat", "mucoïde", "mucoide", "débutant", "debutant", "physiologique",
          "chronique", "ancien", "consolid"])
HIGH_Q = rx(["almost complete", "near complete", "nearly complete", "subtotal", "veća", "veca", "veći", "veci", "visokog stupnja",
             "putpun", "potpuno", "υφολικ", "субтотал", "subtotale", "complete", "full thickness", "full-thickness", "high grade", "high-grade", "moderate", "large", "marked",
             "significant", "extensive", "severe", "massive", "abundant", "grade 3", "grade iii", "grade 4", "grade iv",
             "displaced", "bucket", "total", "acute",
             "complet", "completa", "alto grado", "moderad", "grande", "abundante", "importante", "marcad", "sever", "extens",
             "grado 3", "grado iii", "grado 4", "grado iv", "agud", "desplazad",
             "komplett", "vollständig", "vollstandig", "höhergradig", "hohergradig", "mäßig", "massig", "mittelgradig", "groß",
             "gross", "ausgeprägt", "ausgepragt", "deutlich", "erheblich", "grad 3", "grad iii", "grad 4", "grad iv", "akut",
             "disloziert",
             "tam ", "total", "komplet", "yüksek dereceli", "yuksek dereceli", "orta", "belirgin", "fazla", "büyük", "buyuk",
             "yaygın", "yaygin", "ileri", "grade 3", "grade iii", "grade 4", "grade iv", "evre 3", "evre 4", "akut", "deplase",
             "kompletn", "potpun", "umjeren", "izražen", "izrazen", "obiln", "velik", "znatn", "opsežn", "opsezn", "teš", "iii. stup",
             "iv. stup", "iii°", "iv°", "3. stup", "4. stup", "iii stup", "iv stup", "ii-iv", "iii-iv", "3/4", "3-4", "akutn", "dislocir",
             "πληρη", "πληρουσ", "ολικ", "μετρι", "μεγαλ", "σημαντικ", "εκτεταμεν", "εντον", "βαθμου iii", "βαθμου iv", "βαθμου 3",
             "βαθμου 4", "οξει", "παρεκτοπισμ",
             "пълн", "тотал", "комплет", "умерен", "голям", "значител", "изразен", "обилен", "масивн", "iii степен", "iv степен",
             "3 степен", "4 степен", "остр", "дислоцир",
             "volledig", "compleet", "matig", "groot", "grote", "uitgesproken", "aanzienlijk", "uitgebreid", "fors", "graad 3",
             "graad iii", "graad 4", "graad iv", "acuut", "verplaatst",
             "complet", "complète", "totale", "modéré", "modere", "abondant", "important", "volumineu", "marqué", "étendu", "etendu",
             "sévère", "severe", "grade 3", "grade 4", "stade 3", "stade 4", "aigu", "déplac", "deplac"])
DEGENERATIVE_EDEMA = rx(["degenerat", "subchondral", "subcondral", "arthros", "artros", "dejenerat", "artroz", "εκφυλιστ", "υποχονδρ",
                         "дегенерат", "субхондрал", "geode", "cyst", "quiste", "zyste", "kist", "cist", "κυστ", "кист"])
UNCERTAIN = rx(["?", "suspect", "suspicio", "possib", "probabl", "cannot be excluded", "can not be excluded", "may represent",
                "questionable", "equivocal", "sospecha", "sospech", "posible", "probable", "no se descarta", "dudos", "verdacht",
                "möglich", "moglich", "fraglich", "wahrscheinlich", "nicht auszuschließen", "nicht auszuschliessen", "şüphe", "suphe",
                "olası", "olasi", "muhtemel", "ekarte edilemez", "sumnj", "moguć", "moguc", "vjerojatn", "ddg", "πιθαν", "υποψια",
                "ενδεχομεν", "съмнение", "вероятн", "възможн", "vermoeden", "vermoedelijk", "mogelijk", "waarschijnlijk", "suspicion",
                "probable", "possible", "évoquant", "evoquant", "non exclu"], boundary=False)
EFFUSION_EXCL = rx(["burs", "burz", "baker", "cyst", "cist", "kist", "κυστ", "кист", "popliteal", "poplite", "gastrocnem",
                    "gastroknem", "semimembran", "θυλακ", "бурс", "ganglion"],
                   boundary=False)
CONTUSION_EXCL = rx(["menisc", "menisk", "μηνισκ", "мениск", "ligament", "ligamen", "lig.", "band", "bağ", "bag ", "συνδεσμ",
                     "връзк", "muscle", "muskel", "kas ", "mišić", "misic", "μυ", "мускул", "spier", "soft tissue", "partes blandas",
                     "weichteil", "yumuşak doku", "yumusak doku", "meko tkiv", "μαλακ", "меки тъкани", "weke delen", "tissus mous",
                     "subcutan", "subkutan", "cilt alt", "potkož", "potkoz", "υποδορ", "подкож", "onderhuid", "sous-cutan",
                     "periligament", "parafasyal", "tendon", "tendin", "sehne", "tetiv", "τενοντ", "сухожил", "pees"],
                    boundary=False)


OLD = rx(["old", "healed", "chronic", "antigu", "consolidad", "cronic", "alt", "verheilt", "eski", "iyileş", "iyiles", "stari",
          "zarasl", "παλαι", "πωρωμεν", "стар", "зарастнал", "oud", "geheeld", "ancien", "consolid"])

SPLIT = re.compile(r"[.;\n\r•]+|,(?=\s)|(?<=\s)(?:but|pero|aber|ama|ali|αλλα|но|maar|mais)\s")


SENT = re.compile(r"[.;\n\r•]+")
SUB = re.compile(r",(?=\s)|(?<=\s)(?:but|pero|aber|ama|ali|αλλα|но|maar|mais)\s")
LIST_CONT = re.compile(r"^(?:ni|nor|or|o|noch|oder|niti|ili|ne|veya|ya da|ή|ουτε|или|нито|и|of|ou|y|and|und|en|ve|i|και)\s")


def clauses(report):
    """Cláusulas de cada oración. Una subcláusula sin ninguna estructura anatómica se une a la anterior
    ("LCA..., con pérdida de continuidad"), y en una enumeración negada ("sin derrame, quistes ni Baker")
    los elementos siguientes heredan la negación."""
    out = []
    for sent in SENT.split(norm(report)):
        prev = None
        for sub in (x.strip() for x in SUB.split(sent)):
            if len(sub) <= 2:
                continue
            if prev is not None and not ANY_STRUCT.search(sub):
                sub = prev + ", " + sub
            elif prev is not None and LIST_CONT.match(sub) and (NEGATION_BEFORE.match(prev) or prev.startswith(("no ", "sin ", "without"))):
                sub = prev.split()[0] + " " + sub
            out.append(sub)
            prev = sub
    return out


LOW_STRICT = rx(["partial", "low grade", "low-grade", "mild", "small", "minimal", "minor", "slight", "sprain", "strain",
                 "grade 1", "grade i ", "grade i,", "grade 2", "grade ii", "grade i-ii", "grade 1-2", "intrasubstance", "superficial",
                 "incipient", "early", "beginning", "not reach", "does not extend", "parcial", "bajo grado", "leve", "pequeñ",
                 "pequen", "minim", "discret", "esguince", "grado 1", "grado i ", "grado 2", "grado ii", "incipiente", "inicial",
                 "no alcanza", "no contacta", "partiell", "teilriss", "gering", "klein", "leicht", "diskret", "zerrung", "grad 1",
                 "grad i ", "grad 2", "grad ii", "beginnend", "kısmi", "kismi", "parsiyel", "hafif", "küçük", "kucuk", "grade 1",
                 "grade 2", "grade ii", "grade i ", "evre 1", "evre 2", "erken", "ulaşmayan", "ulasmayan", "parcijaln", "djelomičn",
                 "djelomicn", "blag", "minimal", "početn", "pocetn", "i. stup", "ii. stup", "i/ii", "i°", "ii°", "1. stup", "2. stup",
                 "ne doseže", "ne doseze", "tiny", "sitn", "μερικ", "ηπια", "ηπιο", "μικρη", "μικρο", "ελαχιστ", "βαθμου i ", "βαθμου ii",
                 "βαθμου 1", "βαθμου 2", "αρχομεν", "δεν φθανει", "парциал", "частич", "лек", "малък", "минимал", "i степен",
                 "ii степен", "1 степен", "2 степен", "начал", "не достига", "partieel", "partiële", "partiele", "gering", "klein",
                 "minimaal", "licht", "lichte", "graad 1", "graad i ", "graad 2", "graad ii", "beginnend", "partiel", "partielle",
                 "léger", "leger", "légère", "legere", "petit", "minime", "discret", "entorse", "stade 1", "stade 2", "débutant",
                 "debutant", "n'atteint pas"])


ANY_STRUCT = re.compile("|".join(r.pattern for r in (ACL, MCL, MENISCUS, COMP_MED, COMP_LAT, PF, GENERAL_OA, EFFUSION,
                                                     SYNOVITIS, BAKER, CONTUSION, FRACTURE)))


def qualify(c, m, low=None):
    """Calificación de la mención que empieza en m.start() dentro de la cláusula c."""
    low = low or LOW_Q
    before, after = c[:m.start()], c[m.end():]
    if NEGATION_BEFORE.search(before) or NEGATION_AFTER.search(after) or NEGATION_AFTER.search(before[-25:]):
        return NEG
    if HIGH_Q.search(c):
        return HIGH
    if UNCERTAIN.search(c):
        return LOW
    if low.search(c):
        return LOW
    return PLAIN


def side_hits(c, side_rx, other_rx):
    """La cláusula habla del lado pedido (o de ambos, o de ninguno explícito)."""
    s, o = side_rx.search(c), other_rx.search(c)
    return bool(s) or not o


def score_clause(c):
    """Puntuación por hallazgo para una cláusula; None = el hallazgo no se menciona en ella."""
    out = dict.fromkeys(LABELS)

    def ligament(struct, key):
        if not struct.search(c):
            return
        t = TEAR.search(c)
        if t:
            out[key] = qualify(c, t, LOW_STRICT)
        elif NORMAL.search(c) or NEGATION_BEFORE.search(c) or NEGATION_AFTER.search(c):
            out[key] = NEG

    ligament(ACL, "ACL")
    ligament(MCL, "MCL")

    if MENISCUS.search(c):
        t = TEAR.search(c) or MENISCUS_EXTRA.search(c)
        for key, side, other in (("Medial Meniscus", MEDIAL, LATERAL), ("Lateral Meniscus", LATERAL, MEDIAL)):
            if not side_hits(c, side, other):
                continue
            explicit = bool(side.search(c))
            if t and not explicit:
                if qualify(c, t, LOW_STRICT) == NEG:
                    out[key] = NEG                   # "meniscos sin rotura": niega ambos lados
                continue
            if t:
                q = qualify(c, t, LOW_STRICT)
                if MENISCUS_EXTRA.search(c) and q != NEG:
                    q = HIGH
                out[key] = q
            elif NORMAL.search(c) or NEGATION_BEFORE.search(c) or NEGATION_AFTER.search(c):
                out[key] = NEG
            elif LOW_Q.search(c):
                out[key] = LOW                       # degeneración sin rotura

    oa = OA_PATH.search(c)
    if oa:
        q = qualify(c, oa, LOW_STRICT)
        hits = {"Medial OA": COMP_MED.search(c) or (MEDIAL.search(c) and not PF.search(c)),
                "Lateral OA": COMP_LAT.search(c) or (LATERAL.search(c) and not PF.search(c)),
                "PF OA": PF.search(c)}
        if GENERAL_OA.search(c) and not any(hits.values()):
            hits = dict.fromkeys(hits, True)
            q = min(q, PLAIN) if q != NEG else q
        for key, h in hits.items():
            if h:
                out[key] = q

    for key, term in (("Effusion", EFFUSION), ("Synovitis", SYNOVITIS), ("Baker's", BAKER)):
        ms = list(term.finditer(c))
        if key == "Effusion":                        # líquido en una bursa o en el quiste (±40 caracteres), no en la articulación
            ms = [m for m in ms if not EFFUSION_EXCL.search(w := c[max(0, m.start() - 40):m.end() + 40])
                  or re.search(r"suprapat|subquadri|супрапат|υπερεπιγονατ", w)]   # la bolsa suprapatelar es articular
        if ms:
            qs = [qualify(c, m) for m in ms]
            q = NEG if NEG in qs else max(qs)
            if key == "Effusion" and q != NEG and HEMARTHROS.search(c):
                q = HIGH                             # un hemartros siempre es un derrame importante
            if q != NEG and NORMAL.search(c) and not HIGH_Q.search(c):
                q = NEG                              # "cantidad de líquido normal", "fisiológico"
            if key == "Synovitis" and q == LOW:
                q = PLAIN - .1                       # la sinovitis no tiene umbral de grado en el criterio
            out[key] = q

    f = FRACTURE.search(c)
    if f:
        q = qualify(c, f)
        if q != NEG and OLD.search(c):
            q = LOW
        if q != NEG and q == LOW and not OLD.search(c):
            q = PLAIN                                # "fractura" con un adjetivo débil sigue siendo fractura
        out["Fracture"] = q
    m = CONTUSION.search(c)
    if m and CONTUSION_EXCL.search(c) and not re.search(r"bone|oss|осте|οστ|kemik|kost|knochen|bot|ós", c):
        m = None                                     # edema o "contusión" de partes blandas, menisco o ligamento
    if m:
        q = qualify(c, m)
        if q != NEG and DEGENERATIVE_EDEMA.search(c):
            q = LOW
        elif q == LOW:
            q = PLAIN                                # contusión "pequeña" sigue contando
        out["Contusion"] = q
    return out


def label_report(report):
    best = dict.fromkeys(LABELS, None)
    for c in clauses(report):
        for k, v in score_clause(c).items():
            if v is None:
                continue
            if best[k] is None or (v != NEG and (best[k] == NEG or v > best[k])):
                best[k] = v
    return {k: (NOT_MENTIONED if v is None else v) for k, v in best.items()}


def main():
    os.makedirs(OUT, exist_ok=True)
    train = pd.read_csv(TRAIN_CSV, dtype={"StudyInstanceUID": str})
    rows = [{"StudyInstanceUID": u, **label_report(r)} for u, r in zip(train.StudyInstanceUID, train.Report.fillna(""))]
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/etiquetas_reglas.csv", index=False)
    print(f"→ {OUT}/etiquetas_reglas.csv  {df.shape}")
    print("positivos (≥0,5) por hallazgo, %:")
    print((100 * (df[LABELS] >= .5).mean()).round(1).to_string())
    print("no mencionado, %:")
    print((100 * (df[LABELS] == NOT_MENTIONED).mean()).round(1).to_string())


if __name__ == "__main__":
    main()
