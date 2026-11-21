"""Training corpus, tokenizer, and knowledge passages for Lumo.

Everything here is generated locally — no downloads, no provider APIs.
The corpus is a mix of:
  * synthetic tool-use trajectories (arithmetic, time, memory, knowledge QA)
  * short factual passages that also power the agent's `search` tool
  * chit-chat / identity / capability responses
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #

TOKEN_RE = re.compile(r"[a-z0-9]+|\n|[^a-z0-9\s]")
UNK, BOS, SEP = "<unk>", "<bos>", "<sep>"


class Tokenizer:
    """Simple word/punctuation tokenizer with a fixed vocab."""

    def __init__(self, vocab: list[str]):
        self.itos = list(vocab)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @classmethod
    def train(cls, text: str, cap: int = 14000) -> "Tokenizer":
        counts: dict[str, int] = {}
        for tok in TOKEN_RE.findall(text.lower()):
            counts[tok] = counts.get(tok, 0) + 1
        # Frequency-capped vocab: keep specials first, then most common tokens.
        words = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        vocab = [UNK, BOS, SEP] + [w for w, _ in words[: cap - 3]]
        return cls(vocab)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(t, 0) for t in TOKEN_RE.findall(text.lower())]

    def decode(self, ids) -> str:
        return " ".join(self.itos[int(i)] if 0 <= int(i) < len(self.itos) else UNK for i in ids)


# --------------------------------------------------------------------------- #
# Knowledge passages — offline knowledge base for the `search` tool
# --------------------------------------------------------------------------- #

# (topic keywords, passage text)
PASSAGES: list[tuple[str, str]] = [
    ("sun star fusion", "The Sun is the star at the center of the solar system. It is a nearly perfect sphere of hot plasma, and its core fuses hydrogen into helium, releasing the energy that lightens the Earth. About 1.3 million Earths could fit inside the Sun."),
    ("moon earth orbit", "The Moon is Earth's only natural satellite. It orbits Earth at a mean distance of about 384,400 kilometers and takes roughly 29.5 days to complete one cycle of phases. The Moon's gravity is the main driver of Earth's ocean tides."),
    ("mars red planet", "Mars is the fourth planet from the Sun and is called the red planet because iron oxide dust covers its surface. It has the tallest volcano in the solar system, Olympus Mons, and thin air mostly made of carbon dioxide."),
    ("black holes gravity", "A black hole is a region of spacetime where gravity is so strong that nothing, not even light, can escape. They form when massive stars collapse at the end of their lives. The boundary is called the event horizon."),
    ("light speed vacuum", "Light travels at about 299,792 kilometers per second in a vacuum. Nothing with mass can reach that speed. Because of this, the light we see from the Sun left it about eight minutes and twenty seconds ago."),
    ("gravity force apple", "Gravity is the attraction between objects with mass. Isaac Newton showed that the same force that pulls an apple to the ground keeps the Moon in orbit. Albert Einstein later described gravity as the curvature of spacetime."),
    ("atoms matter element", "Atoms are the basic building blocks of matter. Each atom has a nucleus of protons and neutrons surrounded by electrons. The number of protons, called the atomic number, determines which chemical element the atom belongs to."),
    ("dna genes heredity", "DNA, or deoxyribonucleic acid, is the molecule that carries genetic instructions in living things. It is shaped like a double helix. Genes are segments of DNA that code for proteins and traits passed from parents to offspring."),
    ("photosynthesis plants oxygen", "Photosynthesis is the process plants use to turn sunlight, water, and carbon dioxide into sugar and oxygen. It takes place mainly in chloroplasts that contain chlorophyll. Photosynthesis is the reason Earth's atmosphere has so much oxygen."),
    ("water cycle evaporation", "The water cycle describes how water moves between the ocean, air, and land. Water evaporates, forms clouds, falls as rain or snow, and flows back to the sea. Fresh water is only a small fraction of all water on Earth."),
    ("volcano magma eruption", "A volcano is an opening in Earth's crust through which molten rock, ash, and gases escape. Magma that reaches the surface is called lava. The Ring of Fire around the Pacific Ocean has most of the world's active volcanoes."),
    ("earthquake fault tremor", "An earthquake is the sudden shaking of the ground caused by tectonic plates sliding past each other along faults. Scientists measure earthquakes with the moment magnitude scale. The point underground where a quake starts is the focus."),
    ("hurricane storm tropical", "A hurricane is a large rotating storm that forms over warm tropical ocean water. Sustained winds of at least 119 kilometers per hour define a hurricane. Warm water and humid air give these storms their energy."),
    ("amazon rainforest brazil", "The Amazon rainforest in South America is the largest tropical rainforest on Earth. It contains an enormous share of the world's plant and animal species and produces a large amount of the oxygen in our atmosphere through photosynthesis."),
    ("sahara desert africa", "The Sahara is the largest hot desert in the world, covering much of North Africa. Despite its size, it is not uniformly sandy; large areas are rocky plateau and gravel plain. Temperatures can swing more than thirty degrees in a single day."),
    ("nile river egypt africa", "The Nile is a major river in northeastern Africa flowing north into the Mediterranean Sea. Ancient Egyptian civilization grew along its banks because annual floods deposited fertile silt that made farming possible in a desert climate."),
    ("everest himalaya mountain", "Mount Everest is the highest mountain above sea level, standing at 8,848.62 meters in the Himalayas on the border of Nepal and Tibet. Climbers call its summit the top of the world, and low oxygen makes it dangerous."),
    ("pacific ocean largest", "The Pacific Ocean is the largest and deepest ocean on Earth, covering more area than all the land combined. The Mariana Trench within it contains Challenger Deep, the deepest known point in the ocean, about 11 kilometers down."),
    ("coral reef australia marine", "The Great Barrier Reef off the northeast coast of Australia is the world's largest coral reef system. Coral is built by tiny animals called polyps, and the reef supports thousands of species of fish, turtles, and shellfish."),
    ("honeybees pollination", "Honeybees live in large colonies with a queen, workers, and drones. They collect nectar and pollen, and in doing so they pollinate many crops and wild plants. Bees communicate the location of food with a waggle dance."),
    ("octopus intelligence", "An octopus is a soft-bodied sea animal with eight arms and three hearts. Octopuses are considered very intelligent; they can solve puzzles, open jars, and change both color and texture to blend into their surroundings."),
    ("dolphins mammals ocean", "Dolphins are marine mammals that breathe air and nurse their young. They live in social groups called pods and navigate using echolocation, sending out clicks and listening for the echoes that bounce back from objects."),
    ("wolves pack canidae", "Wolves are wild relatives of dogs that live and hunt in family packs. A pack usually consists of a breeding pair and their offspring. Wolves communicate with howls that can carry for many kilometers across territory."),
    ("pyramids egypt pharaoh", "The Great Pyramids of Giza were built as tombs for Egyptian pharaohs more than four thousand years ago. The largest, built for Khufu, was the tallest human-made structure in the world for over 3,800 years."),
    ("roman empire history rome", "The Roman Empire grew from a city-state on the Italian peninsula into one of the largest empires of the ancient world. Its roads, laws, and Latin language left a lasting mark on Europe and beyond."),
    ("printing press book history", "Johannes Gutenberg's printing press, developed around 1440 in Europe, made it possible to produce books quickly and cheaply. The spread of printed material greatly increased literacy and helped share scientific and political ideas."),
    ("industrial revolution machines", "The Industrial Revolution began in Britain in the eighteenth century when machines like the steam engine transformed manufacturing. It shifted work from farms to factories and changed how most people lived and earned their living."),
    ("united nations peace organization", "The United Nations is an international organization founded in 1945 to promote peace, security, and cooperation among countries. Its headquarters are in New York City, and its member states work together on global problems."),
    ("inflation prices economy", "Inflation is the general rise in prices over time, which reduces the purchasing power of money. Central banks often target a low, steady inflation rate because it encourages spending and investment without eroding savings too quickly."),
    ("compound interest savings", "Compound interest is interest earned on both the original deposit and the interest that has already accumulated. Starting to save early makes a big difference because compounding has more time to grow the balance."),
    ("chess board strategy game", "Chess is a strategy board game for two players on an 8 by 8 grid. Each side begins with sixteen pieces including a king, a queen, rooks, bishops, knights, and pawns. The goal is to checkmate the opponent's king."),
    ("olympics games sport", "The Olympic Games are an international sports festival held every four years, with summer and winter editions. The modern Olympics were revived in 1896, and athletes compete for gold, silver, and bronze medals."),
    ("piano music instrument", "The piano is a musical instrument played by pressing keys that cause hammers to strike strings. It can play many notes at once, which makes it useful for both melody and harmony. Pianos are central to classical, jazz, and pop music."),
    ("coffee bean drink history", "Coffee is a drink made from roasted beans of a plant native to Ethiopia. The beans are ground, brewed with hot water, and drunk worldwide. Caffeine is the natural stimulant that makes coffee popular for alertness."),
    ("chocolate cacao sweet", "Chocolate is made from the beans of the cacao tree, which grows in tropical regions. The beans are fermented, dried, roasted, and ground into cocoa. Cocoa has been used as a drink and food for thousands of years."),
    ("bread baking yeast", "Bread is a staple food made from flour, water, and usually yeast. Yeast ferments the dough, producing carbon dioxide that makes bread rise. Baking then sets the structure and creates the crust through the Maillard reaction."),
    ("sleep health rest", "Sleep is a natural state of rest during which the brain processes memories and the body repairs itself. Most adults need seven to nine hours a night. A regular schedule and a dark, quiet room help improve sleep quality."),
    ("exercise health heart", "Regular physical exercise strengthens the heart, improves circulation, and helps maintain a healthy weight. Activities like brisk walking, cycling, and swimming for about thirty minutes most days bring clear health benefits over time."),
    ("handwashing hygiene disease", "Washing hands with soap and clean water is one of the simplest ways to prevent disease. It removes germs picked up from surfaces and other people. Proper handwashing for about twenty seconds is especially important before eating."),
    ("vaccines immune system", "A vaccine teaches the immune system to recognize a germ without causing the disease. It usually contains a weakened or inactivated form of a pathogen or a piece of one. Vaccination has eliminated or controlled many serious illnesses."),
    ("antibiotics resistance bacteria", "Antibiotics kill bacteria or stop them from growing, which saves lives in serious infections. Taking them when they are not needed, or not finishing a course, lets resistant bacteria survive. Antibiotic resistance is a growing global health problem."),
    ("solar system planets", "The solar system has eight planets orbiting the Sun: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. The four inner planets are rocky, while the four outer planets are gas or ice giants."),
    ("evolution natural selection", "Evolution is the change in living populations over generations. Natural selection favors traits that help organisms survive and reproduce, so those traits become more common. Charles Darwin described this mechanism in On the Origin of Species."),
    ("seasons earth tilt axis", "Earth's seasons are caused by the tilt of its axis, not by its distance from the Sun. When your hemisphere leans toward the Sun you get longer days and summer; leaning away gives shorter days and winter."),
    ("tides moon gravity", "Tides are the rise and fall of sea levels caused mainly by the gravity of the Moon, with some contribution from the Sun. Most coasts see two high tides and two low tides each day as the Earth rotates."),
    ("rainbow sunlight refraction", "A rainbow appears when sunlight is refracted, reflected, and dispersed inside raindrops. The colors we see always run in the order red, orange, yellow, green, blue, indigo, and violet, with red on the outer edge of the arc."),
    ("internet network computers", "The Internet is a global network of computers that exchange data using shared protocols. It grew out of research projects like ARPANET. The World Wide Web, a system of linked pages, is one of the most popular services on it."),
    ("python programming language", "Python is a high-level programming language known for clear syntax and readability. It is widely used for web development, data science, automation, and machine learning. The language is named after Monty Python, not the snake."),
    ("algorithms complexity big o", "An algorithm is a step-by-step procedure for solving a problem. Big O notation describes how the running time or memory needs of an algorithm grow with the input size. Choosing a better algorithm often beats tuning code."),
    ("databases sql tables", "A database stores information in structured tables with rows and columns. SQL is the standard language for querying and updating relational databases. Good indexes make common queries much faster by avoiding full table scans."),
    ("encryption passwords security", "Encryption transforms readable data into unreadable ciphertext that only someone with the right key can reverse. Strong, unique passwords and two-factor authentication protect accounts even if a password list is stolen."),
    ("phishing scams online safety", "Phishing is a scam where attackers send fake messages that look legitimate to trick people into revealing passwords or card numbers. Checking the sender address, avoiding suspicious links, and confirming requests through another channel help prevent it."),
    ("satellites gps navigation", "GPS works by trilateration from satellites that broadcast precise time signals. A receiver compares arrival times to figure out its distance from each satellite, then combines them to determine its exact position on Earth."),
    ("compilers source code", "A compiler translates source code written in a programming language into machine code that a computer can run. Interpreters, by contrast, execute code step by step. Many systems use both, such as just-in-time compilation."),
    ("sound waves hearing", "Sound travels as pressure waves through air, water, and solids. The frequency of the wave determines pitch and the amplitude determines loudness. Human ears typically hear frequencies from about 20 to 20,000 hertz."),
    ("telescope astronomy light", "A telescope gathers and focuses light or other radiation from distant objects. Larger mirrors and lenses can collect more light, letting astronomers see fainter and farther targets. Space telescopes avoid the blurring caused by Earth's atmosphere."),
    ("photos camera lens image", "A camera captures light with a lens that focuses it onto a sensor. The aperture controls how much light enters and the shutter speed controls how long it is exposed. Both settings shape the brightness and motion of a photo."),
    # ── countries & capitals ──────────────────────────────────────────────
    ("france paris capital europe", "France is a country in Western Europe. Its capital is Paris, home of the Eiffel Tower. France is known for art, fashion, cuisine, and wine, and it is one of the most visited countries in the world."),
    ("japan tokyo capital asia", "Japan is an island country in East Asia. Its capital is Tokyo, one of the largest metropolitan areas on Earth. Japan is famous for technology, sushi, cherry blossoms, and a rich tradition of art and literature."),
    ("kenya nairobi capital africa", "Kenya is a country in East Africa. Its capital is Nairobi, which means place of cool waters in the Maasai language. Kenya is known for wildlife safaris, the Great Rift Valley, and long-distance runners."),
    ("brazil brasilia capital south america", "Brazil is the largest country in South America. Its capital is Brasilia, a planned city opened in 1960. Brazil is known for the Amazon rainforest, football, Carnival, and coffee."),
    ("egypt cairo capital africa nile", "Egypt is a country in North Africa. Its capital is Cairo, the largest city in Africa and home to the nearby Great Pyramids of Giza. The Nile River has supported Egyptian life for thousands of years."),
    ("australia canberra capital", "Australia is a country and continent in the Southern Hemisphere. Its capital is Canberra, not Sydney or Melbourne. Australia is known for the Outback, unique wildlife, the Great Barrier Reef, and major cities on its coast."),
    ("canada ottawa capital north america", "Canada is a country in North America spanning six time zones. Its capital is Ottawa. Canada is known for vast forests, maple syrup, hockey, polite cities, and cities like Toronto, Vancouver, and Montreal."),
    ("india new delhi capital asia", "India is a country in South Asia. Its capital is New Delhi. India is the world's most populous country, known for the Taj Mahal, diverse languages, spices, cricket, and a growing technology industry."),
    ("italy rome capital europe", "Italy is a country in Southern Europe shaped like a boot. Its capital is Rome, once the center of the Roman Empire. Italy is famous for art, architecture, pasta, pizza, fashion, and ancient ruins."),
    ("nigeria abuja capital africa", "Nigeria is a country in West Africa and the most populous country on the continent. Its capital is Abuja, while Lagos is its largest city. Nigeria is known for Nollywood films and lively music."),
    # ── chemistry & materials ─────────────────────────────────────────────
    ("water molecule h2o hydrogen oxygen", "Water is made of molecules with two hydrogen atoms and one oxygen atom, written H2O. Water covers about seventy percent of Earth's surface and is essential for every known form of life."),
    ("oxygen gas element breathing", "Oxygen is the chemical element with the symbol O. It is a colorless gas that makes up about twenty-one percent of Earth's atmosphere. Animals breathe oxygen to release energy from food."),
    ("carbon element life diamond", "Carbon is the chemical element with the symbol C. It forms more compounds than any other element and is the backbone of life. In one arrangement carbon forms diamond; in another it forms graphite."),
    ("gold metal element jewelry", "Gold is a soft, shiny metal with the symbol Au. It does not rust or tarnish, so people have used it for jewelry and coins for thousands of years. Gold conducts electricity very well."),
    ("iron rust metal steel", "Iron is a common metal with the symbol Fe. When iron meets oxygen and moisture it forms rust, a flaky reddish coating. Mixing iron with carbon makes steel, which is used in buildings and tools."),
    ("acids bases ph chemistry", "Acids taste sour and have a pH below seven, while bases feel slippery and have a pH above seven. Pure water has a pH of seven. Mixing an acid and a base can neutralize both."),
    # ── human body ────────────────────────────────────────────────────────
    ("heart blood pump organ", "The heart is a muscular organ that pumps blood through the body. It beats about one hundred thousand times a day, sending oxygen and nutrients to tissues and carrying waste away."),
    ("lungs breathing oxygen organs", "The lungs are the organs of breathing. They take in oxygen from the air and release carbon dioxide. The left lung is slightly smaller than the right to make room for the heart."),
    ("brain nervous system organ", "The brain is the control center of the nervous system. It processes senses, controls movement, stores memories, and manages breathing and heartbeat — all using about eighty-six billion nerve cells."),
    ("bones skeleton calcium organ", "An adult skeleton has 206 bones. Bones are living tissue that store calcium, protect organs, and work with muscles to move the body. Bone marrow inside bones makes new blood cells."),
    ("muscles movement protein organ", "There are more than six hundred muscles in the human body. Muscles contract to create movement, hold the skeleton upright, and generate heat that keeps body temperature stable."),
    ("blood vessels circulation red", "Blood travels through a network of vessels: arteries carry blood away from the heart and veins bring it back. Red blood cells carry oxygen using a protein called hemoglobin."),
    # ── inventions & technology history ───────────────────────────────────
    ("wheel invention transportation", "The wheel is one of the oldest inventions, appearing more than five thousand years ago. It was first used for pottery and later for carts and chariots. The wheel transformed transportation and machinery."),
    ("telephone bell invention communication", "Alexander Graham Bell is credited with patenting the telephone in 1876. Early telephones carried voice over wires between homes and offices, shrinking distances long before the internet existed."),
    ("light bulb edison invention", "Joseph Swan and Thomas Edison both developed practical incandescent light bulbs in the 1870s. Electric lighting replaced gas lamps and candles, letting people work and travel safely after dark."),
    ("steam engine industrial revolution", "The steam engine, improved by James Watt, converted heat into motion. It powered factories, mines, and locomotives, and it was the driving force behind the Industrial Revolution."),
    ("paper invention writing china", "Paper was invented in China around two thousand years ago and later spread along trade routes. Cheap paper made books possible, education widespread, and record-keeping far easier than writing on stone or clay."),
    # ── animals extra ─────────────────────────────────────────────────────
    ("giraffe tall animal africa", "The giraffe is the tallest living animal, with adults reaching over five meters. Its long neck has the same seven neck vertebrae as most mammals, just greatly stretched. Giraffes live in African savannas."),
    ("zebra stripes animal africa", "Zebras are African equids famous for their black and white stripes. Each zebra's pattern is unique like a fingerprint. Scientists believe stripes help confuse biting insects and predators in a herd."),
    ("cheetah fast animal speed", "The cheetah is the fastest land animal, accelerating from zero to about one hundred kilometers per hour in a few seconds. It hunts in open grasslands and can only sustain top speed briefly."),
    ("elephant largest animal memory", "Elephants are the largest land animals. They live in tight family herds led by the oldest female, mourn their dead, and can remember water sources and other elephants for decades."),
    ("penguin bird antarctica swim", "Penguins are flightless birds that live mostly in the Southern Hemisphere. Their wings have become flippers for swimming. Emperor penguins endure Antarctic winters by huddling together against the cold."),
    ("bats flying mammal echolocation", "Bats are the only mammals that can truly fly. Many species navigate in the dark using echolocation, sending out squeaks and listening for echoes. There are over one thousand four hundred bat species."),
    # ── space extra ───────────────────────────────────────────────────────
    ("saturn rings planet gas giant", "Saturn is the sixth planet from the Sun and is famous for its bright rings. The rings are made mostly of ice and rock particles. Saturn is a gas giant with dozens of moons."),
    ("jupiter largest planet storm", "Jupiter is the largest planet in the solar system. Its Great Red Spot is a storm larger than Earth that has raged for centuries. Jupiter has a faint ring system and many moons."),
    ("venus hottest planet greenhouse", "Venus is the second planet from the Sun and the hottest, with surface temperatures around four hundred sixty degrees Celsius. A thick carbon dioxide atmosphere traps heat through a runaway greenhouse effect."),
    ("comets meteorites space rocks", "Comets are icy bodies that grow glowing tails as they near the Sun. When pieces of space rock survive the fall through Earth's atmosphere they are called meteorites, some of which are billions of years old."),
    ("international space station orbit", "The International Space Station is a laboratory in low Earth orbit, circling the planet about every ninety minutes. It has been continuously inhabited by astronauts since the year two thousand."),
    # ── computing extra ───────────────────────────────────────────────────
    ("binary code computers bits", "Computers store and process information as binary code, strings of zeros and ones called bits. Eight bits form a byte. Everything on a screen ultimately comes from patterns of these simple switches."),
    ("cpu processor computer chip", "The central processing unit, or CPU, is the brain of a computer. It fetches instructions from memory, decodes them, and performs arithmetic and logic billions of times per second."),
    ("ram memory computer temporary", "RAM is a computer's short-term working memory. It holds programs and data currently in use so the CPU can reach them quickly. When power turns off, the contents of RAM disappear."),
    ("open source software free code", "Open source software publishes its source code so anyone may read, change, and share it. Linux, Python, and many web tools are open source. Large communities improve the code together."),
    ("cookies web browser tracking", "Cookies are small text files that websites store in your browser. They remember logins, cart contents, and preferences. Cookies can also track browsing across sites, which is why privacy laws mention them."),
    # ── music & art ───────────────────────────────────────────────────────
    ("beethoven composer classical music", "Ludwig van Beethoven was a German composer who bridged the Classical and Romantic eras. He wrote nine symphonies, including the famous Fifth, while composing most of them after losing his hearing."),
    ("mozart composer classical prodigy", "Wolfgang Amadeus Mozart was an Austrian composer who performed for European royalty as a child. He composed over six hundred works, including operas, symphonies, and concertos, before his death at thirty-five."),
    ("jazz music origin new orleans", "Jazz originated in New Orleans around the early nineteen hundreds, blending blues, ragtime, and African rhythms. It is known for swing, improvisation, and instruments like the trumpet, saxophone, and double bass."),
    ("hip hop music culture turntables", "Hip hop culture began in the Bronx in the nineteen seventies with DJing, rapping, breakdancing, and graffiti. Turntables became instruments as DJs isolated and extended drum breaks for dancers."),
    ("renaissance art italy painting leonardo vinci michelangelo", "The Renaissance was a revival of art and learning that began in Italy in the fourteen hundreds. Artists like Leonardo da Vinci and Michelangelo used perspective and anatomy to create lifelike masterpieces."),
    # ── sports ────────────────────────────────────────────────────────────
    ("football world cup sport", "The FIFA World Cup is the biggest football tournament, held every four years between national teams. Brazil has won it the most times. The two thousand twenty-two final in Qatar is widely called the greatest ever."),
    ("marathon race sport distance", "A marathon is a forty-two point one nine five kilometer race. It is named after the ancient Greek messenger who ran from Marathon to Athens to announce a victory, then collapsed and died."),
    ("basketball sport invention", "Basketball was invented in eighteen ninety-one by James Naismith in Massachusetts using a peach basket and a soccer ball. It grew into one of the world's most popular indoor sports."),
    ("tennis sport origin lawn", "Modern tennis developed in Britain in the nineteenth century from an older French game. Matches can be singles or doubles, and the four Grand Slam tournaments are the sport's most prestigious events."),
    # ── economics basics ──────────────────────────────────────────────────
    ("supply demand prices economy", "Supply and demand set most market prices. When a product is scarce and wanted, its price rises; when it is plentiful, the price falls. Producers raise output when prices are high."),
    ("barter trade money history", "Before money, people traded goods directly, called bartering. Swapping a chicken for cloth required a double coincidence of wants. Common currencies solved that problem by being a medium of exchange everyone accepts."),
    ("banks money savings loans", "Banks take deposits from savers and lend that money to borrowers at a higher interest rate. The difference pays for staff and systems. Deposit insurance protects small savers if a bank fails."),
    # ── weather & earth ───────────────────────────────────────────────────
    ("thunder lightning storm weather", "Lightning is a massive electric discharge inside clouds or to the ground. It heats the air to around thirty thousand degrees Celsius, which expands explosively and creates the sound we hear as thunder."),
    ("snow ice frozen precipitation", "Snow forms when water vapor freezes into ice crystals high in clouds and falls as flakes. Each flake usually has six branches because of the hexagonal shape of ice crystals."),
    ("monsoon rain season weather", "A monsoon is a seasonal wind shift that brings heavy rains to South and Southeast Asia. The word comes from the Arabic for season. Monsoon rains refill rivers but can also cause floods."),
    ("frost dew cold weather", "Dew forms when objects cool overnight and water vapor in the air condenses onto them. When temperatures drop below freezing, that moisture becomes frost — tiny ice crystals on grass and leaves."),
    # ── everyday science ──────────────────────────────────────────────────
    ("sundial clock time history", "Before mechanical clocks, sundials told time using the Sun's shadow. A gnomon casts a shadow that moves across marked hours. Sundials only work in daylight, so water and sand clocks were used at night."),
    ("libraries books knowledge history", "Libraries have preserved knowledge for millennia, from ancient clay tablets to printed books. The ancient Library of Alexandria was famous for collecting scrolls from across the Mediterranean world."),
    ("museums artifacts culture", "Museums collect, preserve, and display artifacts of art, science, and history. They let the public study objects that would otherwise be private or lost, and they often run research programs too."),
    ("maps projection geography", "Maps flatten the round Earth onto paper, which always distorts something. The Mercator map preserves shapes and angles for navigation but makes polar regions look far larger than they really are."),
    ("refrigerator food cold appliance", "A refrigerator slows bacterial growth by keeping food at about four degrees Celsius. A coolant evaporates inside the coils, absorbing heat from the interior, then releases it outside the appliance."),
]


# --------------------------------------------------------------------------- #
# Knowledge mega-expansion — table-derived passages generated locally.
# Countries & capitals, periodic elements, history timeline, geography,
# computing, math and science. No downloads, no provider APIs.
# --------------------------------------------------------------------------- #

_COUNTRIES: list[tuple[str, str, str]] = [
    ("afghanistan", "kabul", "central asia"), ("albania", "tirana", "europe"),
    ("algeria", "algiers", "north africa"), ("angola", "luanda", "south africa"),
    ("argentina", "buenos aires", "south america"), ("armenia", "yerevan", "west asia"),
    ("australia", "canberra", "oceania"), ("austria", "vienna", "europe"),
    ("azerbaijan", "baku", "west asia"), ("bahamas", "nassau", "caribbean"),
    ("bahrain", "manama", "west asia"), ("bangladesh", "dhaka", "south asia"),
    ("belarus", "minsk", "europe"), ("belgium", "brussels", "europe"),
    ("belize", "belmopan", "central america"), ("benin", "porto-novo", "west africa"),
    ("bhutan", "thimphu", "south asia"), ("bolivia", "sucre", "south america"),
    ("bosnia", "sarajevo", "europe"), ("botswana", "gaborone", "south africa"),
    ("brazil", "brasilia", "south america"), ("bulgaria", "sofia", "europe"),
    ("burkina faso", "ouagadougou", "west africa"), ("burundi", "gitega", "east africa"),
    ("cambodia", "phnom penh", "southeast asia"), ("cameroon", "yaounde", "central africa"),
    ("canada", "ottawa", "north america"), ("chad", "ndjamena", "central africa"),
    ("chile", "santiago", "south america"), ("china", "beijing", "east asia"),
    ("colombia", "bogota", "south america"),
    ("dr congo", "kinshasa", "central africa"), ("costa rica", "san jose", "central america"),
    ("croatia", "zagreb", "europe"), ("cuba", "havana", "caribbean"),
    ("cyprus", "nicosia", "europe"), ("czechia", "prague", "europe"),
    ("denmark", "copenhagen", "europe"), ("djibouti", "djibouti", "east africa"),
    ("dominican republic", "santo domingo", "caribbean"), ("ecuador", "quito", "south america"),
    ("egypt", "cairo", "north africa"), ("el salvador", "san salvador", "central america"),
    ("estonia", "tallinn", "europe"), ("eswatini", "mbabane", "south africa"),
    ("ethiopia", "addis ababa", "east africa"), ("finland", "helsinki", "europe"),
    ("france", "paris", "western europe"), ("gabon", "libreville", "central africa"),
    ("georgia", "tbilisi", "west asia"), ("germany", "berlin", "europe"),
    ("ghana", "accra", "west africa"), ("greece", "athens", "europe"),
    ("guatemala", "guatemala city", "central america"), ("guinea", "conakry", "west africa"),
    ("guyana", "georgetown", "south america"), ("haiti", "port-au-prince", "caribbean"),
    ("hungary", "budapest", "europe"), ("iceland", "reykjavik", "europe"),
    ("india", "new delhi", "south asia"), ("indonesia", "jakarta", "southeast asia"),
    ("iran", "tehran", "west asia"), ("iraq", "baghdad", "west asia"),
    ("ireland", "dublin", "europe"), ("israel", "jerusalem", "west asia"),
    ("italy", "rome", "europe"), ("ivory coast", "yamoussoukro", "west africa"),
    ("jamaica", "kingston", "caribbean"), ("japan", "tokyo", "east asia"),
    ("jordan", "amman", "west asia"), ("kazakhstan", "astana", "central asia"),
    ("kenya", "nairobi", "east africa"), ("kuwait", "kuwait city", "west asia"),
    ("kyrgyzstan", "bishkek", "central asia"), ("laos", "vientiane", "southeast asia"),
    ("latvia", "riga", "europe"), ("lebanon", "beirut", "west asia"),
    ("lesotho", "maseru", "south africa"), ("liberia", "monrovia", "west africa"),
    ("libya", "tripoli", "north africa"), ("lithuania", "vilnius", "europe"),
    ("luxembourg", "luxembourg", "europe"), ("madagascar", "antananarivo", "east africa"),
    ("malawi", "lilongwe", "east africa"), ("malaysia", "kuala lumpur", "southeast asia"),
    ("mali", "bamako", "west africa"), ("mauritania", "nouakchott", "west africa"),
    ("mexico", "mexico city", "north america"), ("moldova", "chisinau", "europe"),
    ("mongolia", "ulaanbaatar", "east asia"), ("montenegro", "podgorica", "europe"),
    ("morocco", "rabat", "north africa"), ("mozambique", "maputo", "east africa"),
    ("myanmar", "naypyidaw", "southeast asia"), ("namibia", "windhoek", "south africa"),
    ("nepal", "kathmandu", "south asia"), ("netherlands", "amsterdam", "europe"),
    ("new zealand", "wellington", "oceania"), ("nicaragua", "managua", "central america"),
    ("niger", "niamey", "west africa"), ("nigeria", "abuja", "west africa"),
    ("north korea", "pyongyang", "east asia"), ("north macedonia", "skopje", "europe"),
    ("norway", "oslo", "europe"), ("oman", "muscat", "west asia"),
    ("pakistan", "islamabad", "south asia"), ("panama", "panama city", "central america"),
    ("papua new guinea", "port moresby", "oceania"), ("paraguay", "asuncion", "south america"),
    ("peru", "lima", "south america"), ("philippines", "manila", "southeast asia"),
    ("poland", "warsaw", "europe"), ("portugal", "lisbon", "europe"),
    ("qatar", "doha", "west asia"), ("romania", "bucharest", "europe"),
    ("russia", "moscow", "east europe"), ("rwanda", "kigali", "east africa"),
    ("saudi arabia", "riyadh", "west asia"), ("senegal", "dakar", "west africa"),
    ("serbia", "belgrade", "europe"), ("sierra leone", "freetown", "west africa"),
    ("slovakia", "bratislava", "europe"), ("slovenia", "ljubljana", "europe"),
    ("somalia", "mogadishu", "east africa"), ("south africa", "pretoria", "south africa"),
    ("south korea", "seoul", "east asia"), ("south sudan", "juba", "east africa"),
    ("spain", "madrid", "europe"), ("sri lanka", "colombo", "south asia"),
    ("sudan", "khartoum", "north africa"), ("suriname", "paramaribo", "south america"),
    ("sweden", "stockholm", "europe"), ("switzerland", "bern", "europe"),
    ("syria", "damascus", "west asia"), ("taiwan", "taipei", "east asia"),
    ("tajikistan", "dushanbe", "central asia"), ("tanzania", "dodoma", "east africa"),
    ("thailand", "bangkok", "southeast asia"), ("togo", "lome", "west africa"),
    ("tunisia", "tunis", "north africa"), ("turkey", "ankara", "west asia"),
    ("turkmenistan", "ashgabat", "central asia"), ("uganda", "kampala", "east africa"),
    ("ukraine", "kyiv", "europe"), ("united arab emirates", "abu dhabi", "west asia"),
    ("united kingdom", "london", "europe"), ("united states", "washington dc", "north america"),
    ("uruguay", "montevideo", "south america"), ("uzbekistan", "tashkent", "central asia"),
    ("venezuela", "caracas", "south america"), ("vietnam", "hanoi", "southeast asia"),
    ("yemen", "sanaa", "west asia"), ("zambia", "lusaka", "east africa"),
    ("zimbabwe", "harare", "east africa"),
]

_ELEMENTS: list[tuple[int, str, str, str]] = [
    (1, "H", "hydrogen", "lightest element"), (2, "He", "helium", "noble gas"),
    (3, "Li", "lithium", "alkali metal"), (4, "Be", "beryllium", "alkaline earth metal"),
    (5, "B", "boron", "metalloid"), (6, "C", "carbon", "nonmetal"),
    (7, "N", "nitrogen", "nonmetal"), (8, "O", "oxygen", "nonmetal"),
    (9, "F", "fluorine", "halogen"), (10, "Ne", "neon", "noble gas"),
    (11, "Na", "sodium", "alkali metal"), (12, "Mg", "magnesium", "alkaline earth metal"),
    (13, "Al", "aluminium", "post-transition metal"), (14, "Si", "silicon", "metalloid"),
    (15, "P", "phosphorus", "nonmetal"), (16, "S", "sulfur", "nonmetal"),
    (17, "Cl", "chlorine", "halogen"), (18, "Ar", "argon", "noble gas"),
    (19, "K", "potassium", "alkali metal"), (20, "Ca", "calcium", "alkaline earth metal"),
    (22, "Ti", "titanium", "transition metal"), (24, "Cr", "chromium", "transition metal"),
    (25, "Mn", "manganese", "transition metal"), (26, "Fe", "iron", "transition metal"),
    (27, "Co", "cobalt", "transition metal"), (28, "Ni", "nickel", "transition metal"),
    (29, "Cu", "copper", "transition metal"), (30, "Zn", "zinc", "transition metal"),
    (31, "Ga", "gallium", "post-transition metal"), (32, "Ge", "germanium", "metalloid"),
    (33, "As", "arsenic", "metalloid"), (34, "Se", "selenium", "nonmetal"),
    (35, "Br", "bromine", "halogen"), (36, "Kr", "krypton", "noble gas"),
    (37, "Rb", "rubidium", "alkali metal"), (38, "Sr", "strontium", "alkaline earth metal"),
    (40, "Zr", "zirconium", "transition metal"), (42, "Mo", "molybdenum", "transition metal"),
    (46, "Pd", "palladium", "transition metal"), (47, "Ag", "silver", "transition metal"),
    (48, "Cd", "cadmium", "transition metal"), (49, "In", "indium", "post-transition metal"),
    (50, "Sn", "tin", "post-transition metal"), (53, "I", "iodine", "halogen"),
    (54, "Xe", "xenon", "noble gas"), (55, "Cs", "caesium", "alkali metal"),
    (56, "Ba", "barium", "alkaline earth metal"), (74, "W", "tungsten", "transition metal"),
    (78, "Pt", "platinum", "transition metal"), (79, "Au", "gold", "transition metal"),
    (80, "Hg", "mercury", "transition metal"), (82, "Pb", "lead", "post-transition metal"),
    (83, "Bi", "bismuth", "post-transition metal"), (92, "U", "uranium", "actinide"),
    (94, "Pu", "plutonium", "actinide"),
]

_HISTORY: list[tuple[int, str]] = [
    (3000, "the first pyramids were built in egypt"), (776, "the first olympic games were held in greece"),
    (509, "the roman republic was founded"), (221, "qin unified china"),
    (476, "the western roman empire fell"), (622, "the islamic calendar began with the hijra"),
    (800, "charlemagne was crowned emperor of the romans"),
    (1066, "the normans conquered england at hastings"), (1215, "the magna carta was sealed at runnymede"),
    (1347, "the black death reached europe"), (1440, "johannes gutenberg developed the printing press"),
    (1453, "constantinople fell to the ottomans"), (1492, "columbus reached the americas"),
    (1503, "leonardo da vinci began painting the mona lisa"), (1517, "martin luther posted his ninety-five theses"),
    (1543, "copernicus published his heliocentric model"), (1600, "shakespeare was at the height of his career"),
    (1607, "jamestown became the first permanent english settlement in america"),
    (1687, "isaac newton published his principia"), (1776, "the united states declared independence"),
    (1789, "the french revolution began with the storming of the bastille"),
    (1804, "napoleon crowned himself emperor of france"), (1815, "the battle of waterloo ended napoleon's rule"),
    (1859, "darwin published on the origin of species"), (1863, "the emancipation proclamation was issued in the us"),
    (1869, "the suez canal opened"), (1876, "alexander graham bell patented the telephone"),
    (1879, "thomas edison demonstrated the incandescent light bulb"),
    (1903, "the wright brothers made the first powered flight"),
    (1905, "albert einstein published the theory of special relativity"),
    (1914, "world war one began"), (1918, "world war one ended"),
    (1920, "women gained the right to vote in the united states"),
    (1928, "alexander fleming discovered penicillin"), (1929, "the wall street crash triggered the great depression"),
    (1936, "the spanish civil war began"), (1939, "world war two began"),
    (1945, "world war two ended and the united nations was founded"),
    (1947, "india gained independence from britain"), (1948, "the state of israel was declared"),
    (1953, "watson and crick described the structure of dna"),
    (1957, "the soviet union launched sputnik, the first satellite"),
    (1961, "yuri gagarin became the first human in space"),
    (1969, "apollo eleven landed the first humans on the moon"),
    (1971, "the first microprocessor was invented at intel"),
    (1989, "the berlin wall fell"), (1991, "the soviet union dissolved"),
    (1990, "the world wide web was proposed by tim berners-lee"),
    (2001, "the wikipedia encyclopedia was launched"),
    (2007, "the first apple iphone was released"),
    (2015, "the paris climate agreement was adopted"),
]

_GEOG: list[tuple[str, str]] = [
    ("nile river africa", "The Nile is a major river in northeastern Africa flowing north into the Mediterranean Sea. At about 6,650 kilometers it is often counted as the longest river in the world, and it has supported Egyptian life for thousands of years."),
    ("amazon river south america", "The Amazon is the largest river by discharge volume on Earth, flowing through Brazil and Peru to the Atlantic Ocean. The Amazon rainforest around it produces a large share of the world's oxygen and hosts unmatched biodiversity."),
    ("mississippi river usa", "The Mississippi River drains much of the central United States, flowing about 3,700 kilometers from Minnesota to the Gulf of Mexico. Mark Twain's stories made it famous in American literature."),
    ("yangtze river china", "The Yangtze is Asia's longest river, flowing about 6,300 kilometers across China to the East China Sea. It powers huge hydroelectric dams and irrigates major farmlands."),
    ("ganga ganges river india", "The Ganges River flows from the Himalayas across northern India and Bangladesh to the sea. It is sacred to Hindus and supports one of the densest river basins on Earth."),
    ("mount everest himalaya", "Mount Everest is the tallest mountain above sea level at 8,849 meters, standing in the Himalayas on the border of Nepal and Tibet. Climbers call its Nepali name Sagarmatha."),
    ("k2 mountain karakoram", "K2 is the second tallest mountain in the world at 8,611 meters, rising in the Karakoram range on the Pakistan-China border. It is considered harder to climb than Everest."),
    ("kilimanjaro tanzania", "Mount Kilimanjaro in Tanzania is Africa's highest peak at 5,895 meters. It is a dormant volcano with three cones and snow near its summit despite sitting near the equator."),
    ("andes mountains south america", "The Andes are the longest continental mountain range, running about 7,000 kilometers down the western edge of South America. They host the Atacama Desert and many active volcanoes."),
    ("rocky mountains north america", "The Rocky Mountains stretch more than 4,800 kilometers through western North America from Canada to New Mexico, forming a major continental divide."),
    ("sahara desert africa", "The Sahara is the largest hot desert in the world, covering about 9.2 million square kilometers of northern Africa. It has sand seas, rocky plateaus and night temperatures near freezing."),
    ("gobi desert asia", "The Gobi is a cold desert spanning southern Mongolia and northern China. Dinosaurs roamed it, and it sits in the rain shadow of the Himalayas."),
    ("antarctica polar", "Antarctica is the coldest, driest and windiest continent, covered by an ice sheet holding most of the world's fresh water. It has no permanent population, only research stations."),
    ("arctic ocean north", "The Arctic Ocean is the smallest and shallowest ocean, centered on the North Pole. Its sea ice melts and refreezes with the seasons and is shrinking as the climate warms."),
    ("great barrier reef australia", "The Great Barrier Reef off northeast Australia is the world's largest coral reef system, visible from space and home to thousands of marine species."),
    ("grand canyon usa", "The Grand Canyon in Arizona was carved by the Colorado River over millions of years. It exposes rock layers nearly two billion years old."),
    ("victoria falls africa", "Victoria Falls on the border of Zambia and Zimbabwe is one of the largest waterfalls in the world by combined width and height, known locally as Mosi-oa-Tunya, the smoke that thunders."),
    ("pacific ocean", "The Pacific Ocean is the largest and deepest ocean, covering about a third of Earth's surface. It contains the Mariana Trench, the deepest known point in the ocean."),
    ("atlantic ocean", "The Atlantic Ocean is the second largest ocean, separating the Americas from Europe and Africa. The mid-Atlantic ridge runs along its floor where new crust forms."),
    ("indian ocean", "The Indian Ocean is the warmest ocean, bounded by Asia, Africa and Australia. Monsoon winds have guided trade across it for centuries."),
]

_COMPUTING: list[tuple[str, str]] = [
    ("binary code computers bits", "Computers store and process information as binary code, strings of zeros and ones called bits. Eight bits form a byte, and all text, images and sound are ultimately encoded as numbers in base two."),
    ("cpu processor computing", "The central processing unit, or CPU, is the computer's brain. It fetches instructions from memory, decodes them and executes arithmetic and logic, billions of times per second."),
    ("ram memory computing", "RAM is a computer's short-term working memory. It is fast but volatile: contents vanish when power is off, which is why documents must be saved to storage."),
    ("hard drive storage", "Hard drives and SSDs hold data long term. Hard drives use spinning magnetic platters, while solid-state drives use flash memory with no moving parts and much faster access."),
    ("internet network computing", "The internet is a global network of networks that exchanges data using standardized protocols. The World Wide Web is one service that runs on top of it."),
    ("http web protocol", "HTTP is the protocol that moves web pages between browsers and servers. HTTPS adds TLS encryption so eavesdroppers cannot read the traffic."),
    ("database storage queries", "A database stores structured data that programs query and update. Relational databases keep tables with keys, while document stores keep flexible records."),
    ("algorithm computing", "An algorithm is a precise finite sequence of steps that solves a problem or performs a computation. Sorting, searching and routing are classic examples."),
    ("compiler programming", "A compiler translates source code written in a programming language into machine code that a computer can execute directly. Interpreters run the code step by step instead."),
    ("operating system computing", "An operating system manages hardware and provides services to programs: memory, processors, files, devices and networking. Linux, Windows and macOS are common examples."),
    ("encryption security", "Encryption transforms readable data into ciphertext that only someone with the right key can unlock. It protects messages, passwords and payments on the internet."),
    ("password security", "A password is a secret string that proves identity. Strong passwords are long and unique, and password managers generate and store them so people do not reuse weak ones."),
    ("wifi network", "Wi-Fi is wireless networking based on the IEEE 802.11 standards. It uses radio waves to link devices to a local network and then to the internet."),
    ("bluetooth devices", "Bluetooth is a short-range wireless protocol for connecting devices like headphones and keyboards. It hops across many radio frequencies to avoid interference."),
    ("cloud computing", "Cloud computing delivers computing resources over the internet on demand. Instead of owning servers, organizations rent storage and compute and pay for what they use."),
    ("artificial intelligence computing", "Artificial intelligence is the field of making machines perform tasks that normally need human intelligence, such as recognizing patterns, understanding language and making decisions."),
    ("neural network computing", "An artificial neural network is a model of connected units inspired by brains. Layers of weights are tuned by gradient descent so the network maps inputs to desired outputs."),
    ("transformer model architecture", "A transformer is a neural network architecture built on self-attention, letting every token weigh every other token. It underlies modern language models and Lumo uses a tiny decoder-only version."),
    ("git version control", "Git is a distributed version control system that records snapshots of a project. Commits form a history, branches isolate work, and remotes share it with others."),
    ("spam filtering computing", "Spam filters classify unwanted email using rules and learned statistics. They look at sender reputation, message content and links to decide what to block."),
]

_MATHFACTS: list[tuple[str, str]] = [
    ("geometry shapes math", "Geometry studies shapes, sizes and positions. A circle's area is pi times the radius squared, a triangle's angles sum to 180 degrees, and the Pythagorean theorem links the sides of right triangles."),
    ("algebra equations math", "Algebra uses symbols and rules to solve for unknowns. Rearranging an equation keeps both sides balanced, which is how formulas are derived and solved."),
    ("prime numbers math", "A prime number is divisible only by one and itself. There are infinitely many primes, and every integer greater than one factors uniquely into primes."),
    ("fractions decimals math", "Fractions express parts of a whole as one number over another. Dividing the top by the bottom converts a fraction to a decimal, and percentages are fractions with denominator one hundred."),
    ("percentages math", "A percentage means per one hundred. To find p percent of x compute x times p divided by 100, and to express a ratio as a percentage multiply it by one hundred."),
    ("statistics average math", "The mean of a set of numbers is their sum divided by the count. The median is the middle value and the mode is the most frequent one."),
    ("probability math", "Probability measures how likely an event is, from zero impossible to one certain. For equally likely outcomes it is the favorable count divided by the total count."),
    ("pythagorean theorem math", "The Pythagorean theorem states that in a right triangle the square of the hypotenuse equals the sum of the squares of the other two sides: a squared plus b squared equals c squared."),
    ("order of operations math", "Order of operations requires parentheses first, then exponents, then multiplication and division from left to right, and finally addition and subtraction from left to right."),
    ("integers number line math", "Integers include the whole numbers and their negatives on the number line. Adding a negative moves left, and subtracting a negative moves right."),
    ("area perimeter math", "Perimeter is the distance around a shape while area measures the surface inside it. A rectangle's perimeter is twice length plus width and its area is length times width."),
    ("volume geometry math", "Volume measures the space a solid occupies. A box is length times width times height, a cylinder is pi r squared times height, and a sphere is four thirds pi r cubed."),
    ("ratios proportions math", "A ratio compares two quantities, and a proportion states that two ratios are equal. Cross multiplication solves for an unknown in a proportion."),
    ("roman numerals math", "Roman numerals use letters for values: I is one, V five, X ten, L fifty, C one hundred, D five hundred and M one thousand, with subtractive pairs like IV for four."),
    ("fibonacci sequence math", "The Fibonacci sequence starts zero and one and each later term is the previous two added: zero, one, one, two, three, five, eight, thirteen. It appears in shells, leaves and spirals."),
]

_SCIENCE: list[tuple[str, str]] = [
    ("gravity physics force", "Gravity is the attraction between masses. Isaac Newton showed it keeps planets in orbit, and Einstein later described it as curved spacetime caused by mass and energy."),
    ("energy physics conservation", "Energy cannot be created or destroyed, only transformed. Kinetic energy is motion, potential energy is stored position, and heat is disordered microscopic motion."),
    ("electricity physics current", "Electric current is the flow of charge, measured in amperes. Voltage pushes the charge and resistance opposes it, related by Ohm's law: voltage equals current times resistance."),
    ("magnetism physics", "Magnets have north and south poles that attract opposite poles and repel like ones. Moving electric charges create magnetic fields, which is how motors and generators work."),
    ("light optics physics", "Light travels in waves and also behaves like particles called photons. It reflects off mirrors, refracts through glass, and the visible spectrum runs from red through violet."),
    ("sound waves physics", "Sound is a pressure wave that needs a medium. Its pitch follows frequency and its loudness follows amplitude, and it travels faster in solids than in air."),
    ("photosynthesis biology", "Photosynthesis lets plants convert sunlight, water and carbon dioxide into glucose and oxygen. It powers most food chains and produces much of the oxygen in the atmosphere."),
    ("cell biology unit of life", "The cell is the basic unit of life. Animal cells lack a wall while plant cells have cellulose walls and chloroplasts, and bacteria cells have no nucleus at all."),
    ("dna genetics biology", "DNA is a double helix of nucleotide bases: adenine pairs with thymine and cytosine with guanine. Genes are DNA segments that code for proteins inherited from parents."),
    ("evolution biology", "Evolution is the change of species over generations through natural selection. Traits that help survival and reproduction become more common in the population."),
    ("human heart anatomy", "The human heart pumps blood with four chambers. The right side sends blood to the lungs and the left side sends oxygen-rich blood around the whole body, beating about one hundred thousand times a day."),
    ("human brain anatomy", "The human brain contains billions of neurons coordinating thought, memory and movement. Its wrinkled cortex handles senses and reasoning while the cerebellum manages balance."),
    ("skeleton human body", "An adult human skeleton has 206 bones that support the body, protect organs and produce blood cells in the marrow. The smallest bone is the stapes in the middle ear."),
    ("digestion human body", "Digestion breaks food into absorbable nutrients. Chewing and stomach acid start the process, enzymes in the small intestine finish it, and the gut lining absorbs the products."),
    ("periodic table chemistry", "The periodic table arranges elements by atomic number into groups and periods. Elements in the same group share similar chemistry, like the noble gases or the alkali metals."),
    ("atoms chemistry", "Atoms are the building blocks of matter with a nucleus of protons and neutrons orbited by electrons. The number of protons defines the element."),
    ("chemical bond chemistry", "Chemical bonds hold atoms together. Ionic bonds transfer electrons between atoms, while covalent bonds share electron pairs, and metallic bonds delocalize electrons across a metal."),
    ("acids bases chemistry", "Acids donate protons in solution and bases accept them. The pH scale from zero to fourteen measures acidity, with seven neutral, and mixing an acid with a base makes a salt and water."),
    ("water chemistry h2o", "Water is a polar molecule of two hydrogens and one oxygen. Its hydrogen bonding gives it high heat capacity, surface tension and the ability to dissolve many salts."),
    ("ozone layer atmosphere", "The ozone layer high in the stratosphere absorbs most of the Sun's ultraviolet radiation. The Montreal Protocol phased out chemicals that were depleting it."),
    ("greenhouse effect climate", "The greenhouse effect warms Earth's surface. Gases like carbon dioxide and methane trap outgoing infrared radiation, and extra human emissions are amplifying the warming."),
    ("earth orbit seasons", "Earth orbits the Sun once a year while tilted on its axis. The tilt, not distance, causes seasons: each hemisphere leans toward the Sun during its summer."),
    ("moon phases astronomy", "The Moon's phases come from its position relative to Earth and Sun. As it orbits, the sunlit half we see changes from new moon through full moon and back."),
    ("solar system astronomy", "The solar system holds the Sun and everything orbiting it: eight planets, dwarf planets, moons, asteroids and comets. The four inner planets are rocky and the outer four are giant."),
    ("stars astronomy fusion", "Stars are balls of plasma fusing hydrogen into helium in their cores. Their mass decides their fate: small ones fade as white dwarfs, massive ones explode as supernovae."),
    ("galaxy astronomy", "A galaxy is a gravitationally bound system of stars, gas and dust. The Milky Way is a barred spiral holding hundreds of billions of stars including our Sun."),
    ("fossil record geology", "Fossils preserve remains or traces of ancient life in rock. They show how species changed over time and help date the layers in which they are found."),
    ("plate tectonics geology", "Earth's outer shell is broken into plates that drift on the mantle. Collisions build mountains, spreading ridges create crust, and sliding faults cause earthquakes."),
    ("volcano geology", "A volcano is an opening where magma, ash and gas escape from below the surface. The Pacific Ring of Fire marks many of the world's most active volcanoes."),
]

# ── extend the passage pool ──────────────────────────────────────────────────
for _c, _cap, _reg in _COUNTRIES:
    PASSAGES.append((
        f"{_c} {_cap} country {_reg}",
        f"{_c.title()} is a country in {_reg}. Its capital is {_cap.title()}. "
        f"It is one of the many nations represented in Lumo's geography knowledge base.",
    ))
for _num, _sym, _name, _cat in _ELEMENTS:
    PASSAGES.append((
        f"element {_name} {_sym} chemistry",
        f"{_name.title()} ({_sym}) is chemical element number {_num}. It is classified as {_cat}. "
        f"Its symbol is {_sym} on the periodic table.",
    ))
for _year, _event in _HISTORY:
    PASSAGES.append((
        f"history {_year} timeline",
        f"In {_year}, {_event}. This event is part of the historical record kept in Lumo's knowledge base.",
    ))
PASSAGES.extend(_GEOG)
PASSAGES.extend(_COMPUTING)
PASSAGES.extend(_MATHFACTS)
PASSAGES.extend(_SCIENCE)


# --------------------------------------------------------------------------- #
# Trajectory templates
# --------------------------------------------------------------------------- #

GREETINGS = ["hi", "hello", "hey", "yo", "good morning", "good afternoon", "good evening", "hi there", "hello there"]

ARITH_SYM = {
    "+": ["plus", "added to", "with"],
    "-": ["minus", "subtracted from", "less than"],
    "*": ["times", "multiplied by", "times as much as"],
    "/": ["divided by", "over"],
}


def _arith_question(a: int, b: int, sym: str, rng: random.Random) -> str:
    style = rng.randrange(6)
    words = ARITH_SYM[sym]
    w = rng.choice(words)
    if style == 0:
        return f"what is {a} {sym} {b} ?"
    if style == 1:
        return f"calculate {a} {sym} {b}"
    if style == 2:
        return f"what does {a} {w} {b} equal ?"
    if style == 3:
        return f"how much is {a} {sym} {b} ?"
    if style == 4:
        return f"compute {a} {sym} {b} please"
    return f"{a} {sym} {b} = ?"


def _safe_arith(a: int, b: int, sym: str) -> float:
    if sym == "+":
        return a + b
    if sym == "-":
        return a - b
    if sym == "*":
        return a * b
    if b == 0:
        return 0.0
    return a / b


def _fmt_num(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.6g}"


def _gen_arithmetic(rng: random.Random, n: int) -> list[str]:
    out = []
    syms = ["+", "-", "*", "/"]
    thoughts = [
        "i will use the calculator tool .",
        "let me compute that with the calculator .",
        "the calculator can handle this arithmetic .",
        "i should calculate the result .",
    ]
    for _ in range(n):
        sym = rng.choice(syms)
        if sym == "+":
            a, b = rng.randrange(1, 999), rng.randrange(1, 999)
        elif sym == "-":
            a, b = rng.randrange(10, 999), rng.randrange(1, 99)
        elif sym == "*":
            a, b = rng.randrange(2, 99), rng.randrange(2, 99)
        else:
            b = rng.randrange(2, 30)
            r = rng.randrange(2, 60)
            a = b * r
        result = _fmt_num(_safe_arith(a, b, sym))
        q = _arith_question(a, b, sym, rng)
        expr = f"{a} {sym} {b}"
        finish_style = rng.randrange(3)
        if finish_style == 0:
            finish = f"{expr} = {result} ."
        elif finish_style == 1:
            finish = f"the answer is {result} ."
        else:
            finish = f"it equals {result} ."
        out.append(
            f"user : {q}\nlumo : {rng.choice(thoughts)}\n"
            f"action : calculator {expr}\nobservation : {result}\n"
            f"lumo : i have the result .\nfinish : {finish}"
        )
    return out


def _gen_drills(rng: random.Random, n: int) -> list[str]:
    """Master-the-drill bank: full multiplication table (1-100), exact
    division, squares and percentages — thousands of distinct facts."""
    out = []
    thoughts = [
        "i will use the calculator tool .",
        "let me compute that with the calculator .",
        "the calculator can handle this arithmetic .",
        "i should calculate the result .",
    ]
    # deterministic pool: every (a, b) pair in 1..100 for *, plus / and %
    pool: list[tuple[int, int, str]] = []
    for a in range(1, 101):
        for b in range(1, 101):
            pool.append((a, b, "*"))
            if a % b == 0:
                pool.append((a, b, "/"))
    for p in range(1, 101, 5):            # percentages 5,10,...,100 of 1..400
        for x in range(1, 401, 7):
            pool.append((p, x, "%"))
    rng.shuffle(pool)
    for i in range(n):
        a, b, kind = pool[i % len(pool)]
        if kind == "*":
            sym, expr, result = "*", f"{a} * {b}", _fmt_num(a * b)
        elif kind == "/":
            sym, expr, result = "/", f"{a} / {b}", _fmt_num(a / b)
        else:
            # p percent of x -> calculator-safe expression
            sym, expr, result = "%", f"{b} * ({a} / 100)", _fmt_num(b * (a / 100))
        if kind == "%":
            q = rng.choice([
                f"what is {a} percent of {b} ?",
                f"how much is {a}% of {b} ?",
                f"find {a} percent of {b}",
            ])
        else:
            q = _arith_question(a, b, sym, rng)
        finish_style = rng.randrange(3)
        if finish_style == 0:
            finish = f"{expr} = {result} ."
        elif finish_style == 1:
            finish = f"the answer is {result} ."
        else:
            finish = f"it equals {result} ."
        out.append(
            f"user : {q}\nlumo : {rng.choice(thoughts)}\n"
            f"action : calculator {expr}\nobservation : {result}\n"
            f"lumo : i have the result .\nfinish : {finish}"
        )
    return out


def _gen_time(rng: random.Random, n: int) -> list[str]:
    out = []
    questions = [
        "what time is it ?",
        "what is the current time ?",
        "tell me the time now",
        "what 's the time right now ?",
        "what is today 's date ?",
        "what date is it today ?",
        "what day is it today ?",
        "tell me today 's date",
        "what 's the date and time ?",
    ]
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    months = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
    thoughts = ["i will check the clock .", "the time tool knows this .", "let me look up the current time ."]
    for _ in range(n):
        q = rng.choice(questions)
        hh, mm, ss = rng.randrange(24), rng.randrange(60), rng.randrange(60)
        day = rng.choice(days)
        d, mo, yr = rng.randrange(1, 29), rng.choice(months), rng.randrange(2024, 2031)
        stamp = f"{yr} - {mo[:3]} - {d:02d} {hh:02d} : {mm:02d} : {ss:02d} utc ( {day} )"
        if "time" in q and "date" not in q:
            finish = f"it is {hh:02d} : {mm:02d} : {ss:02d} utc ."
        elif "day" in q:
            finish = f"today is {day} ."
        else:
            finish = f"today is {day} , {d} {mo} {yr} ."
        out.append(
            f"user : {q}\nlumo : {rng.choice(thoughts)}\n"
            f"action : time now\nobservation : {stamp}\n"
            f"lumo : i have the time .\nfinish : {finish}"
        )
    return out


_THINGS = ["color", "food", "sport", "movie", "city", "animal", "season", "subject", "music band", "hobby"]
_VALUES = {
    "color": ["blue", "red", "green", "purple", "orange", "black", "white", "teal"],
    "food": ["pizza", "sushi", "rice", "pasta", "burger", "curry", "salad", "bread"],
    "sport": ["football", "tennis", "basketball", "running", "swimming", "cricket", "volleyball"],
    "movie": ["titanic", "avatar", "inception", "frozen", "jaws", "matrix"],
    "city": ["nairobi", "mombasa", "paris", "tokyo", "cairo", "london", "lagos", "rome"],
    "animal": ["dog", "cat", "elephant", "lion", "dolphin", "eagle"],
    "season": ["summer", "winter", "spring", "autumn"],
    "subject": ["math", "history", "biology", "physics", "art", "music"],
    "music band": ["coldplay", "beatles", "queen", "imagine dragons"],
    "hobby": ["reading", "gaming", "cooking", "hiking", "drawing", "chess"],
}


def _gen_memory(rng: random.Random, n: int) -> list[str]:
    out = []
    remember_qs = [
        "remember that my {thing} is {val}",
        "please remember my {thing} is {val}",
        "save this : my {thing} is {val}",
        "note down that my {thing} is {val}",
        "remember , my {thing} is {val}",
    ]
    recall_qs = [
        "what is my {thing} ?",
        "do you remember my {thing} ?",
        "what did i tell you about my {thing} ?",
        "can you recall my {thing} ?",
        "what 's my favorite {thing} ?",
    ]
    thoughts_r = ["i will store that in memory .", "let me save this note .", "i should remember this ."]
    thoughts_k = ["i will check my memory .", "let me recall the stored note .", "the memory tool can answer this ."]
    for i in range(n):
        thing = rng.choice(_THINGS)
        val = rng.choice(_VALUES[thing])
        if i % 2 == 0:
            q = rng.choice(remember_qs).format(thing=thing, val=val)
            key = f"favorite {thing}" if rng.random() < 0.6 else thing
            out.append(
                f"user : {q}\nlumo : {rng.choice(thoughts_r)}\n"
                f"action : remember {key} = {val}\nobservation : stored {key} = {val}\n"
                f'lumo : done .\nfinish : i will remember that your {thing} is {val} .'
            )
        else:
            q = rng.choice(recall_qs).format(thing=thing, val=val)
            key = f"favorite {thing}" if rng.random() < 0.6 else thing
            out.append(
                f"user : {q}\nlumo : {rng.choice(thoughts_k)}\n"
                f"action : recall {key}\nobservation : {val}\n"
                f'lumo : i found it .\nfinish : your favorite {thing} is {val} .'
            )
    return out


def _gen_knowledge(rng: random.Random, n: int) -> list[str]:
    out = []
    thoughts = ["i will search my knowledge base .", "let me look that up .", "the search tool should know this ."]
    questions = [
        "tell me about {topic}",
        "what do you know about {topic} ?",
        "explain {topic} to me",
        "why is {topic} important ?",
        "how does {topic} work ?",
        "i want to learn about {topic}",
        "give me facts about {topic}",
    ]
    for _ in range(n):
        keys, text = rng.choice(PASSAGES)
        topic = rng.choice(keys.split())
        # A couple of topic variants from the keyword list
        if rng.random() < 0.4:
            topic = " ".join(keys.split()[:2])
        q = rng.choice(questions).format(topic=topic)
        out.append(
            f"user : {q}\nlumo : {rng.choice(thoughts)}\n"
            f"action : search {topic}\nobservation : {text}\n"
            f"lumo : here is what i found .\nfinish : {text}"
        )
    return out


def _gen_chat(rng: random.Random, n: int) -> list[str]:
    out = []
    direct = [
        ("hi", "hello ! i am lumo , an ai agent . how can i help ?"),
        ("hello", "hello ! i am lumo , an ai agent . how can i help ?"),
        ("hey", "hey ! what can i do for you ?"),
        ("good morning", "good morning ! what would you like to know ?"),
        ("who are you ?", "i am lumo , a small neural agent that runs entirely on your machine ."),
        ("what is your name ?", "my name is lumo . i am an ai agent with tools ."),
        ("what are you ?", "i am lumo , an ai agent built from scratch with a tiny neural model ."),
        ("what can you do ?", "i can calculate , tell the time , remember things , and search my knowledge base ."),
        ("help", "try asking me math , the date and time , to remember a fact , or a knowledge question ."),
        ("what tools do you have ?", "i have calculator , time , remember , recall , and search tools ."),
        ("thanks", "you are welcome !"),
        ("thank you", "you are welcome ! anything else ?"),
        ("thank you so much", "happy to help !"),
        ("bye", "goodbye ! come back any time ."),
        ("goodbye", "see you later !"),
        ("how are you ?", "i am running well and ready to help . what do you need ?"),
    ]
    questions = [q for q, _ in direct]
    # repeat with slight variation to bulk up the set
    for _ in range(n):
        q, a = rng.choice(direct)
        out.append(f"user : {q}\nlumo : i know how to answer this .\nfinish : {a}")
    rng.shuffle(questions)
    return out


def _gen_two_turn(rng: random.Random, n: int) -> list[str]:
    """Multi-turn trajectories: remember→recall, calc→calc, search→search.

    These teach the model to handle a prior exchange before the current
    question — the exact format the agent uses at inference time.
    """
    out = []
    followups = [
        "and what about {q2} ?",
        "then {q2}",
        "now {q2}",
        "okay , {q2}",
    ]
    for i in range(n):
        kind = i % 3
        if kind == 0:
            thing = rng.choice(_THINGS)
            val = rng.choice(_VALUES[thing])
            key = f"favorite {thing}" if rng.random() < 0.6 else thing
            out.append(
                f"user : remember that my {thing} is {val}\n"
                f"lumo : i will store that in memory .\n"
                f"action : remember {key} = {val}\n"
                f"observation : stored {key} = {val}\n"
                f"lumo : done .\n"
                f"finish : i will remember that your {thing} is {val} .\n"
                f"user : what is my {thing} ?\n"
                f"lumo : i will check my memory .\n"
                f"action : recall {key}\n"
                f"observation : {val}\n"
                f"lumo : i found it .\n"
                f"finish : your favorite {thing} is {val} ."
            )
        elif kind == 1:
            a, b = rng.randrange(2, 99), rng.randrange(2, 99)
            c, d = rng.randrange(2, 99), rng.randrange(2, 99)
            r1, r2 = _fmt_num(a * b), _fmt_num(c * d)
            q2 = rng.choice([f"{c} * {d}", f"{c} times {d}", f"{c} multiplied by {d}"])
            out.append(
                f"user : what is {a} * {b} ?\n"
                f"lumo : i will use the calculator tool .\n"
                f"action : calculator {a} * {b}\n"
                f"observation : {r1}\n"
                f"lumo : i have the result .\n"
                f"finish : {a} * {b} = {r1} .\n"
                f"user : {rng.choice(followups).format(q2=q2)}\n"
                f"lumo : i will calculate that too .\n"
                f"action : calculator {c} * {d}\n"
                f"observation : {r2}\n"
                f"lumo : i have this result too .\n"
                f"finish : {c} * {d} = {r2} ."
            )
        else:
            keys, text = rng.choice(PASSAGES)
            topic = rng.choice(keys.split())
            out.append(
                f"user : tell me about {topic}\n"
                f"lumo : i will search my knowledge base .\n"
                f"action : search {topic}\n"
                f"observation : {text}\n"
                f"lumo : here is what i found .\n"
                f"finish : {text}\n"
                f"user : tell me more about {topic}\n"
                f"lumo : i will search again .\n"
                f"action : search {topic}\n"
                f"observation : {text}\n"
                f"lumo : here is more .\n"
                f"finish : {text}"
            )
    return out


def _gen_multistep(rng: random.Random, n: int) -> list[str]:
    """Two calculator calls in a single turn."""
    out = []
    for _ in range(n):
        a, b = rng.randrange(2, 60), rng.randrange(2, 60)
        c, d = rng.randrange(2, 60), rng.randrange(2, 60)
        r1 = _fmt_num(a * b)
        r2 = _fmt_num(c * d)
        q = f"what is {a} times {b} and {c} times {d} ?"
        out.append(
            f"user : {q}\nlumo : i will calculate both parts .\n"
            f"action : calculator {a} * {b}\nobservation : {r1}\n"
            f"lumo : now the second part .\n"
            f"action : calculator {c} * {d}\nobservation : {r2}\n"
            f"lumo : i have both results .\nfinish : {a} times {b} is {r1} and {c} times {d} is {r2} ."
        )
    return out


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

SYS_LINE = "<sys> you are lumo , an ai agent that answers questions with tools . </sys>"


# (from_unit, to_unit) → expression template with {v}
CONVERSIONS: dict[tuple[str, str], str] = {
    ("km", "mi"): "{v} * 0.621371",
    ("mi", "km"): "{v} * 1.609344",
    ("kg", "lb"): "{v} * 2.20462",
    ("lb", "kg"): "{v} * 0.453592",
    ("c", "f"): "{v} * 9 / 5 + 32",
    ("f", "c"): "({v} - 32) * 5 / 9",
    ("m", "ft"): "{v} * 3.28084",
    ("ft", "m"): "{v} * 0.3048",
    ("l", "gal"): "{v} * 0.264172",
    ("g", "oz"): "{v} * 0.035274",
}

UNIT_ALIASES = {
    "km": "km", "kilometer": "km", "kilometers": "km", "kilometre": "km",
    "kilometres": "km", "mi": "mi", "mile": "mi", "miles": "mi",
    "kg": "kg", "kilogram": "kg", "kilograms": "kg", "kilo": "kg",
    "kilos": "kg", "lb": "lb", "pound": "lb", "pounds": "lb", "lbs": "lb",
    "c": "c", "celsius": "c", "centigrade": "c", "f": "f",
    "fahrenheit": "f", "m": "m", "meter": "m", "meters": "m",
    "metre": "m", "metres": "m", "ft": "ft", "foot": "ft", "feet": "ft",
    "l": "l", "liter": "l", "liters": "l", "litre": "l", "litres": "l",
    "gal": "gal", "gallon": "gal", "gallons": "gal", "g": "g",
    "gram": "g", "grams": "g", "oz": "oz", "ounce": "oz", "ounces": "oz",
}

_UNIT_DISPLAY = {
    "km": "km", "mi": "miles", "kg": "kg", "lb": "pounds", "c": "celsius",
    "f": "fahrenheit", "m": "meters", "ft": "feet", "l": "liters",
    "gal": "gallons", "g": "grams", "oz": "ounces",
}


def _gen_conversions(rng: random.Random, n: int) -> list[str]:
    out = []
    pairs = list(CONVERSIONS.items())
    thoughts = [
        "i will use the calculator to convert that .",
        "let me convert the units .",
        "the calculator can handle this conversion .",
    ]
    qs = [
        "convert {v} {u1} to {u2}",
        "{v} {u1} to {u2} please",
        "how many {u2} in {v} {u1} ?",
        "{v} {u1} is how many {u2} ?",
        "what is {v} {u1} in {u2} ?",
    ]
    for _ in range(n):
        (u1, u2), tmpl = rng.choice(pairs)
        v = rng.choice([1, 2, 5, 10, 12, 15, 25, 42, 50, 75, 100, 150, 200])
        if rng.random() < 0.3:
            v = rng.randrange(1, 40) + 0.5
        expr = tmpl.format(v=v)
        try:
            result = _safe_arith_eval(expr)
        except Exception:  # noqa: BLE001
            continue
        r = _fmt_num(result)
        d1, d2 = _UNIT_DISPLAY[u1], _UNIT_DISPLAY[u2]
        q = rng.choice(qs).format(v=_fmt_num(v), u1=d1, u2=d2)
        finish_style = rng.randrange(3)
        if finish_style == 0:
            finish = f"{_fmt_num(v)} {d1} = {r} {d2} ."
        elif finish_style == 1:
            finish = f"it is {r} {d2} ."
        else:
            finish = f"{r} {d2} ."
        out.append(
            f"user : {q}\nlumo : {rng.choice(thoughts)}\n"
            f"action : calculator {expr}\nobservation : {r}\n"
            f"lumo : i have the conversion .\nfinish : {finish}"
        )
    return out


def _safe_arith_eval(expr: str) -> float:
    import ast

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp):
            ops = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
                   ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b}
            return ops[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        raise ValueError("bad expr")

    return float(ev(ast.parse(expr, mode="eval")))


def build_blocks(seed: int = 7, scale: float = 1.0) -> list[str]:
    def s(n: int) -> int:
        return max(50, int(n * scale))

    rng = random.Random(seed)
    blocks: list[str] = [SYS_LINE]
    blocks += _gen_arithmetic(rng, s(4000))
    blocks += _gen_drills(rng, s(4500))
    blocks += _gen_multistep(rng, s(600))
    blocks += _gen_time(rng, s(600))
    blocks += _gen_memory(rng, s(1500))
    blocks += _gen_knowledge(rng, s(7000))
    blocks += _gen_chat(rng, s(1000))
    blocks += _gen_two_turn(rng, s(2000))
    blocks += _gen_conversions(rng, s(1500))
    rng.shuffle(blocks)
    return blocks


def build_corpus(seed: int = 7) -> str:
    return ("\n" + SYS_LINE + "\n").join(build_blocks(seed)) + "\n"


@dataclass
class CorpusBundle:
    text: str
    tokens: np.ndarray
    tokenizer: Tokenizer
    starts: np.ndarray  # token offsets where each trajectory begins


def build_bundle(cap: int = 14000, seed: int = 7) -> CorpusBundle:
    blocks = build_blocks(seed)
    text = ("\n" + SYS_LINE + "\n").join(blocks) + "\n"
    tok = Tokenizer.train(text, cap=cap)
    ids = tok.encode(text)
    # trajectory start offsets: block i starts after (block i-1 + separator)
    sep_ids = tok.encode("\n" + SYS_LINE + "\n")
    starts: list[int] = []
    off = 0
    for i, blk in enumerate(blocks):
        if i > 0:
            off += len(sep_ids)
        starts.append(off)
        off += len(tok.encode(blk))
    return CorpusBundle(text=text, tokens=np.asarray(ids, dtype=np.int32),
                        tokenizer=tok,
                        starts=np.asarray(sorted(set(starts)), dtype=np.int64))


def build_val_bundle(tokenizer: Tokenizer, cap: int = 4500) -> CorpusBundle:
    """Held-out trajectories (different RNG seed, shared tokenizer) used to
    measure validation loss — a true generalization signal, since every
    question/number/passage combination in it was never trained on."""
    blocks = build_blocks(99, scale=0.35)
    text = ("\n" + SYS_LINE + "\n").join(blocks) + "\n"
    ids = tokenizer.encode(text)
    sep_ids = tokenizer.encode("\n" + SYS_LINE + "\n")
    starts: list[int] = []
    off = 0
    for i, blk in enumerate(blocks):
        if i > 0:
            off += len(sep_ids)
        starts.append(off)
        off += len(tokenizer.encode(blk))
    return CorpusBundle(text=text, tokens=np.asarray(ids, dtype=np.int32),
                        tokenizer=tokenizer,
                        starts=np.asarray(sorted(set(starts)), dtype=np.int64))


if __name__ == "__main__":
    b = build_bundle()
    print(f"corpus chars : {len(b.text):,}")
    print(f"corpus tokens: {b.tokens.size:,}")
    print(f"vocab        : {b.tokenizer.vocab_size:,}")
