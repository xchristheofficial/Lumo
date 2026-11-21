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

# ── expansion wave 2: people, animals, inventions, arts, sports, food,
#    language, space, health, finance, glossary ──────────────────────────────

_PEOPLE: list[tuple[str, str]] = [
    ("einstein albert physics relativity", "Albert Einstein was a German-born physicist who developed the theory of relativity and the famous equation E equals m c squared. He won the Nobel Prize in 1921 and was born in 1879."),
    ("newton isaac physics gravity", "Sir Isaac Newton was an English mathematician and physicist who formulated the laws of motion and universal gravitation. He was born in 1643 and also invented calculus."),
    ("galileo galilei astronomy telescope", "Galileo Galilei was an Italian astronomer who championed the heliocentric model and improved the telescope. He was born in 1564 and is called the father of observational astronomy."),
    ("curie marie radioactivity nobel", "Marie Curie was a Polish-born physicist and chemist who discovered radioactivity, polonium and radium. She won two Nobel Prizes and was born in 1867."),
    ("darwin charles evolution origin species", "Charles Darwin was an English naturalist who proposed evolution by natural selection after his voyage on HMS Beagle. He published On the Origin of Species in 1859."),
    ("tesla nikola electricity ac motor", "Nikola Tesla was a Serbian-American inventor who championed alternating current electricity and invented the AC induction motor. He was born in 1856."),
    ("edison thomas invention light bulb", "Thomas Edison was an American inventor held to hold 1,093 patents, including the practical light bulb and the phonograph. He was born in 1847."),
    ("lovelace ada programming algorithm", "Ada Lovelace was an English mathematician who wrote the first intended computer program for Charles Babbage's Analytical Engine. She is often called the first programmer."),
    ("turing alan computer science enigma", "Alan Turing was an English mathematician who helped break the Enigma code in World War Two and formalized computation with the Turing machine. He was born in 1912."),
    ("shannon claude information theory", "Claude Shannon was an American mathematician who founded information theory in 1948, showing how much data a message carries. His work underpins all modern digital communication."),
    ("tesla nikola radio tower", "Nikola Tesla demonstrated radio-controlled boats and envisioned wireless power. Wardenclyffe Tower was his ambitious attempt at global wireless transmission."),
    ("faraday michael electromagnetism", "Michael Faraday was an English scientist who discovered electromagnetic induction, the principle behind generators and transformers. He had little formal schooling but changed physics."),
    ("maxwell james electromagnetism", "James Clerk Maxwell was a Scottish physicist whose equations unified electricity, magnetism and light into one theory. He was born in 1831."),
    ("bell alexander telephone", "Alexander Graham Bell patented the telephone in 1876 and helped found the company that became AT&T. He was born in 1847 in Scotland."),
    ("wright brothers aviation first flight", "Orville and Wilwright Wilbur Wright built and flew the first powered airplane at Kitty Hawk in 1903, changing transport forever."),
    ("archimedes mathematics buoyancy", "Archimedes was an ancient Greek mathematician and engineer who discovered the principle of buoyancy and calculated pi with great accuracy. He lived in Syracuse around 287 BC."),
    ("pythagoras theorem mathematics", "Pythagoras was an ancient Greek mathematician credited with the theorem about right triangles: a squared plus b squared equals c squared. He founded a school in Croton."),
    ("euclid geometry elements", "Euclid was an ancient Greek mathematician whose book Elements defined geometry for two thousand years. He worked in Alexandria around 300 BC."),
    ("aristotle philosophy logic", "Aristotle was a Greek philosopher, student of Plato and tutor of Alexander the Great. He founded logic, biology and political philosophy as formal studies."),
    ("plato philosophy academy", "Plato was a Greek philosopher who founded the Academy in Athens and wrote dialogues featuring Socrates. He was born around 427 BC."),
    ("socrates philosophy questioning", "Socrates was an Athenian philosopher famous for the Socratic method of relentless questioning. He was tried and sentenced to death in 399 BC."),
    ("confucius china philosophy", "Confucius was a Chinese philosopher whose teachings on ethics, family and good government shaped East Asian culture for over two thousand years. He lived around 500 BC."),
    ("buddha gautama enlightenment", "Gautama Buddha was an Indian sage whose teachings founded Buddhism, emphasizing compassion and the end of suffering. He lived around the fifth century BC."),
    ("gandhi mahatma independence india", "Mahatma Gandhi led India's independence movement through nonviolent civil disobedience. He was born in 1869 and called his philosophy satyagraha."),
    ("mandela nelson apartheid south africa", "Nelson Mandela fought apartheid in South Africa, spent 27 years in prison and became president in 1994. He won the Nobel Peace Prize in 1993."),
    ("lincoln abraham usa president", "Abraham Lincoln was the sixteenth president of the United States who preserved the Union and ended slavery. He was born in 1809 and assassinated in 1865."),
    ("king martin luther civil rights", "Martin Luther King Jr. led the American civil rights movement with speeches like I Have a Dream. He won the Nobel Peace Prize in 1964."),
    ("napoleon bonaparte france empire", "Napoleon Bonaparte rose from artillery officer to Emperor of France, remaking European law and borders. He was born in 1769 in Corsica."),
    ("cleopatra egypt queen", "Cleopatra was the last active ruler of Ptolemaic Egypt, famous for her alliances with Julius Caesar and Mark Antony. She lived from 69 to 30 BC."),
    ("caesar julius rome", "Julius Caesar was a Roman general and dictator whose assassination in 44 BC helped end the Roman Republic. He crossed the Rubicon in 49 BC."),
    ("alexander the great macedon", "Alexander the Great of Macedon built one of the largest empires in history by age thirty, spreading Greek culture from Egypt to India. He was born in 356 BC."),
    ("victoria queen british empire", "Queen Victoria ruled the British Empire for sixty-three years from 1837 to 1901, giving her name to the Victorian era. She was born in 1819."),
    ("churchill winston wartime prime minister", "Winston Churchill led Britain as prime minister through World War Two and won the Nobel Prize in Literature in 1953. He was born in 1874."),
    ("shakespeare william playwright", "William Shakespeare was an English playwright who wrote Hamlet, Macbeth and Romeo and Juliet. He was born in 1564 in Stratford-upon-Avon."),
    ("beethoven ludwig composer deafness", "Ludwig van Beethoven was a German composer who kept writing masterpieces as he went deaf, including his Ninth Symphony. He was born in 1770."),
    ("bach johann composer baroque", "Johann Sebastian Bach was a German composer of the Baroque era, writing hundreds of cantatas and the Well-Tempered Clavier. He was born in 1685."),
    ("mozart wolfgang prodigy composer", "Wolfgang Amadeus Mozart was an Austrian prodigy who composed over six hundred works before dying at thirty-five in 1791. He was born in 1756."),
    ("van gogh vincent painter stars", "Vincent van Gogh was a Dutch post-impressionist painter famous for The Starry Night and his bold brushwork. He sold only one painting while alive, born in 1853."),
    ("picasso pablo cubism art", "Pablo Picasso was a Spanish painter who co-founded Cubism and produced tens of thousands of works. He was born in 1881 and died in 1973."),
    ("kahlo frida self portraits mexico", "Frida Kahlo was a Mexican painter known for vivid self-portraits exploring pain and identity. She was born in 1907 and married Diego Rivera."),
    ("tolstoy leo war and peace", "Leo Tolstoy was a Russian writer of War and Peace and Anna Karenina, also a moral philosopher. He was born in 1828."),
    ("twain mark huck finn america", "Mark Twain was the pen name of Samuel Clemens, American author of The Adventures of Huckleberry Finn. He was born in 1835 and called the father of American literature."),
    ("austen jane pride prejudice", "Jane Austen was an English novelist known for Pride and Prejudice and Sense and Sensibility, writing about manners and marriage. She was born in 1775."),
    ("orwell george nineteen eighty four", "George Orwell was the pen name of Eric Blair, author of Nineteen Eighty-Four and Animal Farm, writing about totalitarianism. He was born in 1903."),
    ("hawking stephen cosmology brief history", "Stephen Hawking was a British physicist who studied space-time and wrote A Brief History of Time. He was born in 1942 and lived with ALS for decades."),
    ("sagan carl cosmos astronomy", "Carl Sagan was an American astronomer who hosted the Cosmos series and argued for the Voyager Golden Record. He was born in 1934."),
    ("feynman richard physics diagrams", "Richard Feynman was an American physicist known for Feynman diagrams and quantum electrodynamics, also a famous teacher. He won the Nobel Prize in 1965."),
    ("bohr niels atomic model", "Niels Bohr was a Danish physicist who proposed the quantized model of the atom in 1913 and won the Nobel Prize in 1922."),
    ("planck max quantum theory", "Max Planck was a German physicist who founded quantum theory in 1900 by introducing the energy quantum. The Planck constant bears his name."),
    ("heisenberg werner uncertainty principle", "Werner Heisenberg was a German physicist who stated the uncertainty principle: you cannot know position and momentum both exactly. He won the Nobel Prize in 1932."),
    ("fleming alexander penicillin", "Alexander Fleming was a Scottish bacteriologist who discovered penicillin in 1928, the first widely used antibiotic. He won the Nobel Prize in 1945."),
    ("watson crick dna double helix", "Watson and Crick described the double-helix structure of DNA in 1953 at Cambridge, building on Rosalind Franklin's X-ray images."),
    ("volta battery electricity", "Alessandro Volta invented the first chemical battery, the voltaic pile, in 1800. The unit of electric potential, the volt, is named after him."),
    ("pasteur louis germs vaccination", "Louis Pasteur was a French chemist who proved germs cause disease, pasteurized milk and developed vaccines for rabies. He was born in 1822."),
]

_ANIMALS: list[tuple[str, str]] = [
    ("lion big cat africa", "Lions are big cats that live in prides on the African savanna. A lion's roar can be heard up to eight kilometers away, and males grow a thick mane."),
    ("tiger striped cat asia", "Tigers are the largest cat species, with orange coats and black stripes unique to each individual. They swim well and live across Asia."),
    ("elephant largest land animal", "Elephants are the largest living land animals. African elephants have bigger ears than Asian ones, and they use their trunks to drink, smell and greet."),
    ("zebra stripes africa", "Zebras are African horses with black and white stripes. Each pattern is unique, and scientists think stripes help confuse biting flies."),
    ("hippopotamus river africa", "Hippopotamuses are heavy mammals that spend most of the day in rivers and lakes. Despite their size they can run faster than a human on land."),
    ("rhinoceros horn", "Rhinoceroses are armored herbivores with horns made of keratin. Several species are critically endangered because of poaching."),
    ("crocodile reptile ambush", "Crocodiles are large reptiles that wait underwater to ambush prey. They have the strongest measured bite of any animal and cannot stick their tongues out."),
    ("snake reptile shed skin", "Snakes are legless reptiles that sense vibrations and smell with their tongues. They shed their skin as they grow and some are venomous."),
    ("eagle bird of prey", "Eagles are large birds of prey with sharp talons and excellent eyesight. The bald eagle is the national bird of the United States."),
    ("owl nocturnal bird silent flight", "Owls are nocturnal birds that fly almost silently thanks to specialized feathers. Their eyes face forward and they can rotate their heads far around."),
    ("penguin flightless bird antarctic", "Penguins are flightless seabirds of the Southern Hemisphere. They swim with wing-like flippers, huddle for warmth, and the emperor penguin dives very deep."),
    ("dolphin mammal intelligence", "Dolphins are clever marine mammals that breathe air, communicate with whistles and hunt in teams. The bottlenose dolphin is the best known species."),
    ("blue whale largest animal", "The blue whale is the largest animal ever known, reaching thirty meters long. It eats tiny krill and its heart is the size of a small car."),
    ("great white shark predator", "The great white shark is a large predatory fish with rows of replaceable teeth. It can sense electrical fields and breaches the water when hunting seals."),
    ("octopus eight arms intelligence", "Octopuses are mollusks with eight arms, three hearts and blue blood. They solve puzzles, open jars and squirt ink to escape predators."),
    ("jellyfish ancient drifter", "Jellices and jellyfish are simple drifting animals with stinging tentacles. Some species glow in the dark and the immortal jellyfish can revert to youth."),
    ("seahorse upright fish", "Seahorses are small fish that swim upright and curl their tails around seagrass. The male carries the babies in a brood pouch until they hatch."),
    ("ant colony insect strength", "Ants are social insects that live in colonies with queens and workers. They lift many times their own weight and communicate with chemicals called pheromones."),
    ("bee pollination honey", "Bees pollinate flowers while collecting nectar to make honey. A colony has one queen, thousands of workers and dances to tell others where flowers are."),
    ("spider web silk", "Spiders are arachnids with eight legs that spin silk webs to catch prey. Spider silk is stronger than steel by weight and some spiders glide on it."),
    ("butterfly metamorphosis", "Butterflies undergo complete metamorphosis from egg to caterpillar to chrysalis to adult. Their proboscis unrolls to sip nectar and wings are covered in tiny scales."),
    ("mosquito disease vector", "Mosquitoes are flying insects whose bites itch because females drink blood for their eggs. Some species transmit malaria and other diseases."),
    ("termite colonies wood", "Termites are social insects that digest wood with help from microbes in their guts. Their mounds can be meters tall and they recycle dead trees in nature."),
    ("ostrich largest bird", "The ostrich is the largest living bird and cannot fly but runs fast. It has the biggest eyes of any land vertebrate and males display black and white plumage."),
    ("kangaroo marsupial australia", "Kangaroos are Australian marsupials that hop on strong hind legs and carry joeys in a pouch. A red kangaroo can bound at high speed across the outback."),
    ("koala eucalyptus australia", "Koalas are marsupials that sleep up to twenty hours a day eating eucalyptus leaves. They carry babies on their back and have fingerprints like humans."),
    ("giant panda bamboo china", "Giant pandas are bears native to China that eat almost only bamboo. Despite their plant diet they retain a carnivore digestive system and black-and-white coats."),
    ("grizzly bear north america", "Grizzly bears are large North American brown bears that fish for salmon and hibernate in winter. A mother can weigh over two hundred kilograms."),
    ("wolf pack canid", "Wolves are wild canids that hunt in coordinated packs and communicate with howls. They helped inspire the domestic dog and roam forests and tundra."),
    ("fox red clever canid", "Red foxes are adaptable canids found on every continent except Antarctica. They listen for rodents under snow and pounce with a characteristic leap."),
    ("deer antlers", "Deer are hoofed mammals where males usually grow antlers each year. Antlers are the fastest-growing tissue of any mammal and are shed and regrown."),
    ("rabbit burrow mammal", "Rabbits are small mammals that live in burrows and thump warnings when danger approaches. Their front teeth grow continuously throughout life."),
    ("horse domestication speed", "Horses were domesticated around 4,000 years ago and transformed travel and farming. A galloping horse can exceed seventy kilometers per hour."),
    ("camel desert adaptation", "Camels store fat in humps, not water, and can drink huge amounts at once. They tolerate heat and sandstorms, carrying loads across deserts."),
    ("llama andes pack animal", "Llamas are Andean camels used as pack animals. They hum to communicate, spit when annoyed and live in herds high in the mountains."),
    ("goat climbing sure footed", "Goats are sure-footed ruminants that climb steep terrain to browse leaves. They were among the first domesticated animals, raised for milk, wool and meat."),
    ("chicken domestic bird", "Chickens are domesticated junglefowl with over twenty billion alive today. A hen can lay hundreds of eggs a year and roosters crow at dawn."),
    ("flamingo pink flock", "Flamingos are wading birds that get their pink color from carotenoid pigments in shrimp and algae. They feed upside down with specialized bills."),
    ("peacock display feathers", "Peafowl are large birds where males trail iridescent tail feathers in a fan-shaped display to attract females. The train can be longer than the body."),
    ("parrot mimicry intelligence", "Parrots are colorful birds that mimic sounds and can learn to use human words. The African grey parrot is famous for vocabulary and problem solving."),
    ("salmon migration spawning", "Salmon are fish that hatch in freshwater, grow in the ocean and return up rivers to spawn. Their amazing homing journey uses Earth's magnetic field."),
    ("crab sideways shell", "Crabs are crustaceans with hard shells and claws that usually walk sideways. Hermit crabs reuse empty shells and some live deep in the ocean."),
    ("lobster long antennae", "Lobsters are large crustaceans with long antennae and powerful claws. They can live very long, grow by molting and taste with small feet on their claws."),
    ("snail shell spiral", "Snails carry coiled shells and glide on a muscular foot leaving a slime trail. Their shells grow with them and garden snails hibernate in cold weather."),
    ("earthworm soil aerator", "Earthworms tunnel through soil, digesting organic matter and leaving nutrient-rich castings. Charles Darwin studied them and called them nature's ploughs."),
    ("scorpion desert arachnid", "Scorpions are arachnids with pincers and a curved sting. They glow ultraviolet light and can survive months without food in deserts."),
    ("bat mammal echolocation", "Bats are the only mammals that truly fly. Many use echolocation, emitting squeaks and hearing echoes to catch insects in total darkness."),
    ("hedgehog spines", "Hedgehogs are small mammals covered in stiff spines. When threatened they curl into a ball, and they eat insects, snails and garden pests."),
    ("dinosaur extinction fossil", "Dinosaurs dominated Earth for over 160 million years until an asteroid ended the Cretaceous period sixty-six million years ago. Birds are their living descendants."),
]

_INVENTIONS: list[tuple[str, str]] = [
    ("steam engine industrial revolution", "The steam engine converted heat into motion and powered the Industrial Revolution. James Watt's improved design in the 1760s made it efficient enough for factories."),
    ("airplane wright flight transport", "The airplane flies using fixed wings and engines. The Wright brothers' 1903 flight at Kitty Hawk grew into global aviation carrying billions of passengers a year."),
    ("camera photography invention", "A camera captures light on film or a sensor to make photographs. The first permanent photograph was made by Joseph Nicéphore Niépce in 1826."),
    ("battery voltaic pile energy", "A battery converts chemical energy into electric current. Alessandro Volta's 1800 voltaic pile was the first, and modern batteries power phones and cars."),
    ("electric motor spinning force", "An electric motor turns electricity into rotational motion using magnetic fields. It powers appliances, fans, trains and electric vehicles."),
    ("compass navigation magnetism", "The compass is a magnetized needle that points toward Earth's magnetic poles. Mariners used it to navigate oceans from the twelfth century onward."),
    ("thermometer temperature measuring", "A thermometer measures temperature. Galileo and others built early versions, and Daniel Fahrenheit and Anders Celsius defined the scales we still use."),
    ("microbiology microscope tiny life", "A microscope magnifies very small things. Antonie van Leeuwenhoek observed bacteria in the 1670s, founding microbiology."),
    ("bicycle two wheels transport", "The bicycle is an efficient two-wheeled machine powered by pedaling. Modern safety bicycles appeared in the 1880s and remain one of the most efficient vehicles ever."),
    ("automobile car engine", "The automobile combines an internal combustion engine with four wheels. Karl Benz built the first practical motor car in 1885, and Ford later made it affordable."),
    ("locomotive railway train", "The locomotive pulls trains along rails. George Stephenson's Rocket won a famous trial in 1829 and railways transformed trade and travel."),
    ("radio wireless communication", "Radio transmits sound through electromagnetic waves. Guglielmo Marconi sent the first transatlantic radio signal in 1901, broadcasting news and music worldwide."),
    ("television moving pictures", "Television broadcasts moving images and sound into homes. Early mechanical and electronic systems spread in the 1920s and 1930s, reshaping culture."),
    ("transistor switching electronics", "The transistor, invented at Bell Labs in 1947, amplifies and switches signals. It replaced vacuum tubes and made modern electronics possible."),
    ("gps satellite navigation", "GPS uses a constellation of satellites to pinpoint your position anywhere on Earth. Originally military, it now guides phones, ships, planes and farming."),
    ("laser focused light", "A laser produces a narrow beam of coherent light by stimulating atoms. It reads discs, cuts materials, performs eye surgery and carries internet traffic."),
    ("solar panel photovoltaic", "Solar panels convert sunlight directly into electricity using photovoltaic cells. Costs have fallen dramatically, making solar a major source of clean power."),
    ("windmill turbine energy", "Windmills harness moving air to grind grain or generate electricity. Modern wind turbines feed hundreds of gigawatts of renewable power to grids."),
    ("mechanical clock timekeeping", "The mechanical clock measured time with falling weights and escapements from the fourteenth century. It synchronized work, prayer and later railways."),
    ("parachute safe descent", "A parachute slows a fall by creating drag with a large canopy. Leonardo sketched one, and modern skydivers rely on both irreplaceable main and reserve chutes."),
    ("elevator vertical transport", "The elevator made tall buildings practical. Elisha Otis demonstrated a safety brake in 1854, and skyscrapers without lifts would be unthinkable today."),
    ("plastic synthetic material", "Plastics are synthetic polymers molded into endless products. Bakelite was the first fully synthetic plastic in 1907, and recycling is now a major global effort."),
    ("paper writing material", "Paper was invented in China around 100 CE from plant fibers. It replaced bulky scrolls and made books, money and bureaucracy cheap and portable."),
    ("gunpowder invention warfare", "Gunpowder was invented in China around the ninth century. It transformed warfare, mining and celebration, powering fireworks and cannons alike."),
    ("vaccination immunity smallpox", "Vaccination trains the immune system with a safe weakened germ. Edward Jenner's 1796 smallpox vaccine led to the disease's global eradication in 1980."),
    ("sewing machine textile", "The sewing machine stitches fabric quickly with a moving needle. Elias Howe and Isaac Singer popularized it in the nineteenth century, changing clothing production."),
    ("printing press movable type", "Johannes Gutenberg's printing press with movable type around 1440 made books affordable and spread literacy and ideas across Europe."),
    ("telescope astronomy distant light", "A telescope gathers light to see distant objects. Galileo's 1609 observations of moons and craters transformed astronomy and our view of the cosmos."),
    ("microscope invisible world", "A microscope magnifies objects too small for the naked eye. Compound microscopes revealed cells in 1665 and germs that cause disease."),
    ("water wheel hydropower", "Water wheels have used flowing water to grind grain and power machinery for over two thousand years, a precursor to modern hydroelectric turbines."),
]

_ARTS: list[tuple[str, str]] = [
    ("hamlet shakespeare play", "Hamlet is a tragedy by William Shakespeare about the Prince of Denmark avenging his father's death. To be or not to be is its most famous soliloquy."),
    ("nineteen eighty four orwell novel", "Nineteen Eighty-Four by George Orwell depicts a totalitarian state under Big Brother, inventing terms like doublethink and thoughtcrime. It was published in 1949."),
    ("pride and prejudice austen novel", "Pride and Prejudice by Jane Austen opens with the famous line about a man in possession of fortune wanting a wife. It follows Elizabeth Bennet and Mr Darcy."),
    ("odyssey homer epic poem", "The Odyssey is an ancient Greek epic by Homer following Odysseus's ten-year voyage home from the Trojan War, facing Cyclops, sirens and suitors."),
    ("don quixote cervantes novel", "Don Quixote by Miguel de Cervantes, published in 1605, follows a knight who tilts at windmills believing they are giants. It is called the first modern novel."),
    ("great gatsby fitzgerald novel", "The Great Gatsby by F. Scott Fitzgerald captures the Jazz Age through Jay Gatsby's lavish parties and longing for Daisy. It was published in 1925."),
    ("harry potter rowling books", "Harry Potter is a seven-book fantasy series by J. K. Rowling about a young wizard at Hogwarts School. The first book appeared in 1997."),
    ("lord of the rings tolkien", "The Lord of the Rings by J. R. R. Tolkien is an epic fantasy quest to destroy the One Ring. It shaped modern fantasy literature and gaming."),
    ("starry night van gogh painting", "The Starry Night, painted by Vincent van Gogh in 1889, shows a swirling night sky over a quiet village. It hangs in the Museum of Modern Art in New York."),
    ("scream munch painting", "The Scream by Edvard Munch, painted in the 1890s, shows a figure clutching its face under an orange sky. It is an icon of modern anxiety."),
    ("birth of venus botticelli painting", "The Birth of Venus by Sandro Botticelli, around 1485, shows the goddess arriving on a shell. It is a masterpiece of the Italian Renaissance."),
    ("beatles rock band", "The Beatles were a British band from Liverpool featuring Lennon, McCartney, Harrison and Starr. They revolutionized pop music in the 1960s with albums like Sgt Pepper."),
    ("moonlight sonata beethoven", "The Moonlight Sonata is Beethoven's piano sonata of 1801, famous for its dreamlike first movement. He dedicated it to a student, Countess Giulietta Guicciardi."),
    ("ninth symphony beethoven ode to joy", "Beethoven's Ninth Symphony, completed in 1824, ends with the Ode to Joy choral theme. It was written when he was almost totally deaf."),
    ("star wars film franchise", "Star Wars is a film franchise created by George Lucas beginning in 1977, following Jedi heroes against the Empire. Its theme and lightsabers are cultural icons."),
    ("oscar academy award film", "An Oscar is an Academy Award for film achievement, a gold statuette first awarded in 1929. The ceremony in Hollywood is watched worldwide."),
    ("grammy music award", "Grammy Awards honor achievements in music, given by the Recording Academy since 1959. They cover genres from pop and rock to jazz and classical."),
    ("mona lisa leonardo painting", "The Mona Lisa by Leonardo da Vinci is the world's most famous portrait, known for her enigmatic smile. It hangs behind glass in the Louvre, Paris."),
    ("thinker rodin sculpture", "The Thinker by Auguste Rodin depicts a man seated in deep thought. Originally part of The Gates of Hell, it became a symbol of philosophy itself."),
    ("swan lake ballet tchaikovsky", "Swan Lake is a ballet with music by Pyotr Tchaikovsky, first performed in 1877. It tells of a princess turned into a swan and her prince."),
    ("mozart magic flute opera", "The Magic Flute is Mozart's opera of 1791 with the Queen of the Night's dazzling aria. It blends comedy, fairy tale and masonic symbolism."),
    ("hollywood film industry usa", "Hollywood is the neighborhood of Los Angeles that became the center of the American film industry. Movies made there have shaped global entertainment for a century."),
    ("met museum art new york", "The Metropolitan Museum of Art in New York holds thousands of works spanning five thousand years of human creativity from every continent."),
    ("louvre museum paris", "The Louvre in Paris is the world's most visited museum, home to the Mona Lisa and the Venus de Milo. It was once a royal palace."),
    ("shakespeare globe theatre", "The Globe Theatre in London was where many of Shakespeare's plays were first performed. The modern reconstruction stands near the original site on Bankside."),
    ("opera vocal theatre", "Opera combines singing, orchestra and drama on stage. It began in Italy around 1600 and includes works by Verdi, Puccini and Mozart."),
    ("jazz music origin america", "Jazz originated in African American communities of New Orleans with improvisation and swing. Louis Armstrong and Duke Ellington became its legends."),
    ("hip hop music culture", "Hip hop is a culture of rapping, DJing, breakdancing and graffiti that began in New York in the 1970s. It is now one of the world's most popular music styles."),
]

_SPORTS: list[tuple[str, str]] = [
    ("soccer football rules pitch", "Association football, called soccer in some countries, is played by two teams of eleven trying to score in a goal. It is the world's most popular sport with a FIFA World Cup every four years."),
    ("basketball rules hoops", "Basketball is played by two teams shooting a ball through hoops. James Naismith invented it in 1891, and the NBA finals are watched globally."),
    ("tennis rules court rally", "Tennis is a racket sport where players rally a ball over a net. Grand Slams are the four biggest tournaments: Wimbledon, Roland Garros, the US Open and Australia."),
    ("cricket rules wickets", "Cricket is a bat-and-ball game between two sides taking turns batting and bowling. Test matches can last five days, and the World Cup is a global event."),
    ("rugby rules oval ball", "Rugby union is played with an oval ball by two teams of fifteen passing backwards and trying to ground the ball for a try. It began at Rugby School in England."),
    ("baseball rules diamond", "Baseball is played on a diamond where a batter hits a pitch and runs four bases. Nine innings decide it, and the World Series crowns the MLB champion."),
    ("ice hockey rules rink", "Ice hockey is a fast team sport on ice where players skate with sticks and shoot a puck. The NHL is its top professional league."),
    ("golf rules holes clubs", "Golf is played by hitting a ball into holes in as few strokes as possible over eighteen holes. The Masters, Open and PGA are among its majors."),
    ("boxing rules ring", "Boxing is a combat sport where two fighters score with punches in a ring. Weight classes keep matches fair and it has been an Olympic sport since 1904."),
    ("swimming rules races pool", "Competitive swimming races swimmers across pools in freestyle, breaststroke, backstroke and butterfly. Michael Phelps won twenty-three Olympic gold medals."),
    ("marathon distance running", "A marathon is a forty-two point one nine five kilometer race named after the ancient Greek run from Marathon to Athens. Elite runners finish in just over two hours."),
    ("cycling tour de france", "Cycling races include the Tour de France, three weeks of stages across France. Riders climb mountains and sprint for the yellow jersey of the overall leader."),
    ("volleyball rules net", "Volleyball is a team sport where players hit a ball over a net without letting it touch their court. Beach volleyball is an Olympic discipline on sand."),
    ("table tennis ping pong", "Table tennis is a fast racket sport played on a small table with a light ball. It is popular worldwide and an Olympic sport since 1988."),
    ("badminton rules shuttle", "Badminton is played with rackets and a shuttlecock over a high net. Smashes can exceed four hundred kilometers per hour, making it the fastest racket sport."),
    ("figure skating artistry", "Figure skating combines athletic jumps and spins with artistic performance on ice. Skaters are judged on technical elements and program components."),
    ("alpine skiing race", "Alpine skiing races skiers down gated slopes at high speed. Events include slalom, giant slalom, super-G and downhill, all Olympic classics."),
    ("chess strategy board game", "Chess is a strategy board game for two players with sixteen pieces each. Magnus Carlsen has been a dominant world champion, and engines now beat all humans."),
    ("billiards pool snooker", "Billiards covers cue sports like pool and snooker where players strike balls into pockets or snooker opponents behind colors."),
    ("surfing ocean waves", "Surfing rides breaking waves on a board, originating in Polynesia. The WSL tour crowns world champions and it became an Olympic sport in Tokyo 2020."),
    ("skateboarding street park", "Skateboarding rides a board with wheels over streets and parks. It entered the Olympics in 2020 with street and park disciplines."),
    ("archery target bow", "Archery is shooting arrows at a target, an ancient hunting skill now a precision sport. Great Britain and South Korea are traditional Olympic powers."),
    ("fencing sword sports", "Fencing is a duel of three blade types: foil, epee and saber. Fencers score with touches and it has been in the Olympics since the first modern games."),
    ("athletics field events", "Track and field includes sprints, hurdles, relays, jumps and throws. Usain Bolt holds the hundred-meter world record at nine point five eight seconds."),
]

_FOOD: list[tuple[str, str]] = [
    ("bread baking flour", "Bread is made from flour, water and yeast baked into a staple food. Flatbreads are among the oldest prepared foods, and leavened loaves rise with gas from yeast."),
    ("rice staple grain asia", "Rice is the staple grain for more than half of humanity, grown in flooded paddies. White rice is milled while brown rice keeps its bran and germ."),
    ("pasta italy noodles", "Pasta is Italian noodles made from wheat dough and water, dried or fresh. Shapes like spaghetti, penne and rigatoni each hold sauces differently."),
    ("pizza italy cheese tomato", "Pizza is a baked flatbread with tomato sauce and cheese, originating in Naples. The Margherita with basil mirrors the Italian flag's colors."),
    ("sushi japan rice seafood", "Sushi is Japanese vinegared rice with seafood, vegetables or egg. It began as a method of preserving fish and is now served worldwide."),
    ("curry spices stew", "Curry refers to spiced saucy dishes from India, Thailand, Malaysia and beyond. Spice blends like garam masala give each region its signature flavor."),
    ("coffee caffeinated drink", "Coffee is a brewed drink from roasted beans, enjoyed worldwide. It grows on bushes in tropical highlands and caffeine improves alertness."),
    ("tea leaves brew culture", "Tea is made by steeping leaves of the camellia plant. Green, black, white and oolong differ in oxidation, and tea culture spans from Britain to Japan."),
    ("chocolate cacao sweet", "Chocolate comes from cacao beans fermented, roasted and ground. The Aztecs drank it bitter, and conching in the 1870s created smooth modern chocolate."),
    ("cheese milk fermentation", "Cheese is made by curdling milk and aging the solids. Thousands of varieties exist, from fresh mozzarella to sharp aged cheddar and blue-veined Roquefort."),
    ("honey bee product sweet", "Honey is made by bees from nectar and stored in comb. It never spoils if kept sealed, and bees must visit millions of flowers for a kilogram."),
    ("olive oil mediterranean", "Olives are pressed to make olive oil, a cornerstone of Mediterranean cooking. Extra virgin oil comes from the first cold pressing and is rich in antioxidants."),
    ("salt sodium chloride seasoning", "Salt preserves food and enhances flavor. Humans have mined and evaporated it for millennia, and iodized salt prevents thyroid disease."),
    ("spices pepper cinnamon flavor", "Spices like black pepper, cinnamon and cardamom come from plants and once cost fortunes. The spice trade drove exploration and shaped world history."),
    ("banana tropical fruit", "Bananas are curved yellow fruits grown in tropical plantations. They are one of the most eaten fruits and the plants are actually giant herbs."),
    ("apple orchard fruit", "Apples grow on orchard trees and come in thousands of varieties. Johnny Appleseed helped spread orchards across America and crisp textures vary by type."),
    ("potato tuber crop", "Potatoes are starchy tubers first domesticated in the Andes. They fueled population growth in Europe after their arrival from the Americas in the sixteenth century."),
    ("tomato fruit vegetable", "Tomatoes are red fruits used as vegetables, native to western South America. Italians embraced them in cooking after their arrival in Europe four hundred years ago."),
    ("onion garlic allium flavor", "Onions and garlic belong to the allium family and form the flavor base of countless dishes. Cutting them releases compounds that make eyes water."),
    ("noodles asia wheat", "Noodles are long strips of dough eaten across Asia with a history of thousands of years. Chinese legend credits their invention to a Han dynasty chef."),
    ("egg breakfast protein", "Eggs are versatile protein packages laid by hens. A single egg contains everything needed to develop a chick, and whites and yolks cook at different temperatures."),
    ("wine grapes fermentation", "Wine is fermented grape juice, made for over eight thousand years. Red, white and rosé differ in grape skin contact, and regions like Bordeaux are famous."),
    ("beer malt hops fermentation", "Beer is made by fermenting malted grain with hops for bitterness. Ancient civilizations brewed it, and styles range from light lagers to dark stouts."),
]

_LANGUAGE: list[tuple[str, str]] = [
    ("noun part of speech", "A noun is a word naming a person, place, thing or idea. Proper nouns name specific items like London while common nouns name general ones like city."),
    ("verb action word", "A verb expresses an action, occurrence or state of being. Verbs change form for tense and person, like run, ran and will run."),
    ("adjective description word", "An adjective describes a noun, telling which, what kind or how many. Big, blue and thirty are all adjectives that add detail to writing."),
    ("adverb modifies verb", "An adverb modifies a verb, adjective or other adverb, often ending in ly. Quickly, very and tomorrow are adverbs of manner, degree and time."),
    ("pronoun replaces noun", "A pronoun stands in for a noun like he, she, it or they. Pronouns avoid repetition and change form by case: I, me and mine."),
    ("preposition relationship word", "A preposition shows relationship between a noun and the rest of the sentence, like in, on, at and between. It often begins a prepositional phrase."),
    ("conjoining conjunction words", "A conjunction joins words, phrases or clauses: and, but, or, because and although are common examples that glue sentences together."),
    ("tenses past present future", "Verb tenses locate actions in time. Present says she walks, past says she walked and future says she will walk, with perfect forms combining them."),
    ("punctuation marks writing", "Punctuation marks structure writing: periods end statements, commas separate items, question marks end questions and apostrophes show possession."),
    ("sentence subject predicate", "A sentence has a subject doing something and a predicate telling what it does. A complete sentence expresses a thought and ends with punctuation."),
    ("synonym words same meaning", "A synonym is a word with the same meaning as another, like big and large. Synonyms add variety to writing but can differ in tone and register."),
    ("antonym opposite meaning", "An antonym is a word with the opposite meaning, like hot and cold. Pairs like these help learners build vocabulary through contrast."),
    ("idiom figurative expression", "An idiom means something different from its literal words, like break a leg for good luck. Idioms are cultural and often confuse language learners."),
    ("metaphor figure of speech", "A metaphor says one thing is another without like or as, as in time is money. It creates vivid images by direct comparison."),
    ("simile comparison like as", "A simile compares using like or as, as busy as a bee. Unlike metaphors it states the comparison openly."),
    ("alliteration repeated sounds", "Alliteration repeats initial consonant sounds, like Peter Piper picked. It appears in poetry, names and slogans for musical effect."),
    ("grammar language rules", "Grammar is the set of rules governing how words combine into correct sentences. Learning grammar helps both speaking and writing clearly."),
    ("alphabet letters writing", "An alphabet is a set of letters representing sounds. English uses twenty-six letters, while other languages use scripts like Cyrillic, Arabic and Hangul."),
]

_SPACE: list[tuple[str, str]] = [
    ("mercury planet closest sun", "Mercury is the smallest planet and the closest to the Sun, with extreme temperature swings. One Mercurian year lasts only eighty-eight Earth days."),
    ("venus planet hottest greenhouse", "Venus is the hottest planet thanks to a runaway greenhouse atmosphere, even though Mercury is closer to the Sun. It rotates backwards very slowly."),
    ("mars planet red rover", "Mars is the red planet with polar ice caps, volcanoes and canyons. Rovers like Curiosity and Perseverance search for signs of ancient water and life."),
    ("jupiter largest planet storm", "Jupiter is the largest planet, a gas giant with the Great Red Spot storm bigger than Earth. It has dozens of moons including Ganymede."),
    ("saturn rings ice rock", "Saturn is famous for its bright rings of ice and rock particles. It is a gas giant less dense than water and has the moon Titan with lakes of methane."),
    ("uranus ice giant tilt", "Uranus is an ice giant that rotates on its side, likely knocked over by an ancient impact. Its pale blue color comes from methane in the atmosphere."),
    ("neptune distant windy planet", "Neptune is the farthest planet, a deep blue ice giant with the fastest winds in the solar system. It was found by mathematics before being seen."),
    ("asteroid belt rocky bodies", "The asteroid belt lies between Mars and Jupiter, containing millions of rocky bodies. Ceres is the largest, once classified as a planet."),
    ("comet icy visitor", "Comets are icy bodies that grow glowing tails near the Sun. Halley's Comet returns roughly every seventy-six years and was recorded for centuries."),
    ("meteor shooting star", "A meteor is the streak of light from a space rock burning up in the atmosphere. Meteor showers occur when Earth crosses debris trails of comets."),
    ("solar eclipse moon shadow", "A solar eclipse happens when the Moon passes between Earth and the Sun, casting a shadow. Totality briefly turns day into night along a narrow path."),
    ("lunar eclipse blood moon", "A lunar eclipse occurs when Earth's shadow falls on the Moon, often turning it reddish. It can be seen from the entire night side of Earth."),
    ("constellation star patterns", "Constellations are named patterns of stars like Orion and the Big Dipper. Astronomers use them as coordinate regions across the sky."),
    ("light year distance space", "A light year is the distance light travels in one year, about nine point five trillion kilometers. It measures vast interstellar distances."),
    ("international space station", "The International Space Station is a laboratory orbiting Earth where astronauts from many countries live and do science in microgravity."),
    ("hubble space telescope", "The Hubble Space Telescope, launched in 1990, orbits above the atmosphere to capture sharp images of galaxies and helped measure the age of the universe."),
    ("apollo moon missions nasa", "The Apollo program landed twelve astronauts on the Moon between 1969 and 1972. Apollo Eleven first put Neil Armstrong and Buzz Aldrin on the surface."),
    ("mars rover curiosity perseverance", "Mars rovers explore the red surface autonomously, analyzing rocks and sending data home. They have found ancient lakebeds and organic molecules."),
    ("satellite orbiting device", "A satellite is an object orbiting a planet. Hundreds of artificial satellites provide communication, weather forecasts, navigation and Earth observation."),
    ("orbit falling around planet", "An orbit is a curved path kept by falling around a planet while moving sideways fast. Satellites and the Moon are in constant free fall around Earth."),
    ("eclipse astronomy shadow event", "Eclipses occur when one celestial body blocks another. Solar and lunar eclipses let people see orbital mechanics with their own eyes."),
    ("exoplanet other star world", "An exoplanet orbits a star other than the Sun. Thousands have been found since the 1990s, some in habitable zones where liquid water could exist."),
    ("space suit astronaut life support", "A space suit keeps astronauts alive with oxygen, pressure and temperature control. Modern suits let crews work outside stations on spacewalks."),
    ("rocket propulsion launch", "Rockets push forward by expelling mass at high speed, working even in the vacuum of space. Multi-stage designs reach orbit efficiently."),
]

_HEALTH: list[tuple[str, str]] = [
    ("sleep rest health memory", "Sleep restores the body and consolidates memory. Adults generally need seven to nine hours, and deep sleep helps repair tissue and the immune system."),
    ("exercise fitness health", "Regular exercise strengthens the heart, muscles and mood. Recommendations include aerobic activity plus strength training each week for most adults."),
    ("nutrition balanced diet health", "A balanced diet includes fruits, vegetables, grains, protein and healthy fats. Variety across food groups covers most vitamin and mineral needs."),
    ("vitamin c immune fruit", "Vitamin C supports immune function and collagen production. Citrus fruits, peppers and berries are rich sources, and deficiency causes scurvy."),
    ("protein muscle repair", "Proteins are made of amino acids that build and repair tissue. Meat, beans, eggs and dairy provide protein essential for muscles and enzymes."),
    ("water hydration body", "Water makes up most of the body and regulates temperature and transport. Drinking regularly prevents dehydration, especially during heat and exercise."),
    ("handwashing disease prevention", "Washing hands with soap removes germs that spread illness. It is one of the simplest and most effective public health measures known."),
    ("vaccine immune protection", "Vaccines train the immune system to recognize germs safely. They have eradicated smallpox and control measles, polio and many other diseases."),
    ("immune system defense body", "The immune system defends against germs with barriers, white blood cells and memory. It remembers invaders so later responses are faster."),
    ("stress mental health", "Chronic stress harms sleep, mood and the heart. Breathing exercises, exercise, connection and rest all help manage stress levels."),
    ("eyesight vision eye health", "Eyes focus light onto the retina to form images. Screen breaks, good light and eye checks protect vision, and glasses correct focusing errors."),
    ("dental health teeth brushing", "Teeth are protected by brushing twice daily and flossing. Sugar feeds bacteria that cause cavities, and regular dental visits catch problems early."),
    ("calories energy food", "Calories measure the energy food provides. Balancing calories eaten with those used maintains body weight, and nutrient quality matters beyond count."),
    ("allergy immune overreaction", "An allergy is an immune reaction to harmless things like pollen or peanuts. Symptoms range from sneezing to severe anaphylaxis requiring urgent care."),
]

_FINANCE: list[tuple[str, str]] = [
    ("budget personal finance plan", "A budget tracks income and spending so money goes where you intend. Common methods allocate needs, wants and savings from each paycheck."),
    ("saving interest compound growth", "Saving puts money aside for later, and compound interest makes it grow faster over time. Starting early dramatically increases the final amount."),
    ("inflation rising prices", "Inflation is the general rise in prices over time, which reduces purchasing power. Central banks target low steady inflation around two percent."),
    ("tax government revenue", "Taxes fund public services like roads, schools and defense. Income tax, sales tax and value added tax are common types around the world."),
    ("stock market shares company", "The stock market trades shares of companies. Prices reflect expectations about future profits, and indexes like the S and P five hundred track broad performance."),
    ("insurance risk protection", "Insurance spreads risk by collecting premiums to pay for covered losses. Health, home and car insurance protect against large unexpected costs."),
    ("loan borrowing repayment", "A loan is money borrowed and repaid with interest. The interest rate and term determine the total cost of borrowing over the life of the loan."),
    ("credit card debt spending", "A credit card lets you borrow up to a limit for purchases. Balances not paid in full accrue interest, making careful use important for finances."),
    ("currency exchange money", "Currency exchange converts one country's money to another at a market rate. Rates fluctuate and affect travel, trade and investment across borders."),
    ("gdp economy output", "Gross domestic product measures the total value of goods and services a country produces. It is the most cited gauge of economic size and growth."),
    ("unemployment jobs economy", "Unemployment is the share of workers without jobs actively seeking work. It rises in recessions and falls in booms, influencing wages and policy."),
    ("compound interest finance", "Compound interest is interest earned on interest. A simple example: money doubling every ten years at about seven percent annual return."),
]

_GLOSSARY: list[tuple[str, str]] = [
    ("democracy definition government", "Democracy is a system of government where citizens choose leaders by voting. The word comes from Greek demos meaning people and kratos meaning rule."),
    ("republic government state", "A republic is a state where the people hold sovereignty and elect representatives, usually with a head of state who is not a monarch."),
    ("law rules society justice", "Laws are rules made and enforced by authorities to keep order and justice. Breaking laws can lead to penalties, and constitutions define the highest rules."),
    ("philosophy wisdom thinking", "Philosophy is the study of fundamental questions about knowledge, reality, ethics and existence. It uses reason and argument rather than experiment."),
    ("logic valid reasoning", "Logic is the study of valid reasoning. Deductive arguments conclude necessarily from premises while inductive ones generalize from evidence."),
    ("hypothesis scientific guess testable", "A hypothesis is a testable explanation or prediction of a science experiment. Evidence that fails to support it can lead to revising or rejecting it."),
    ("molecule atoms bonded", "A molecule is two or more atoms bonded together, like water made of two hydrogens and one oxygen. Molecules can be tiny or enormous polymers."),
    ("electron negative particle", "An electron is a negatively charged subatomic particle orbiting the nucleus. Electric current is the flow of electrons through a conductor."),
    ("proton nucleus positive", "Protons are positively charged particles in the nucleus. The number of protons defines the element, and isotopes vary by neutron count."),
    ("neutron nucleus neutral", "Neutrons are neutral particles in the atomic nucleus. They hold nuclei together, and free neutrons decay with a half-life of about ten minutes."),
    ("gene dna unit heredity", "A gene is a segment of DNA that codes for a trait or protein. Genes are inherited from parents and variations explain differences between individuals."),
    ("virus microscopic infectious", "A virus is a tiny infectious particle of genetic material inside a protein coat. It only reproduces inside host cells, which antibiotics cannot treat."),
    ("bacteria single celled microbe", "Bacteria are single-celled microbes found almost everywhere. Most are harmless or helpful, while a few cause disease treatable with antibiotics."),
    ("ecosystem living environment", "An ecosystem is a community of organisms interacting with their environment. Energy flows through it as food chains and cycles recycle nutrients."),
    ("biodiversity variety of life", "Biodiversity is the variety of life on Earth at all levels. It supports ecosystems services like pollination, clean water and stable climates."),
    ("climate long term weather", "Climate is the long-term pattern of weather in a region, usually averaged over thirty years. It differs from weather, which changes day to day."),
    ("earthquake ground shaking", "An earthquake is the shaking of the ground when tectonic plates suddenly slip. The Richter scale measures magnitude and seismographs record the waves."),
    ("mineral natural crystal", "A mineral is a naturally occurring inorganic crystal with a definite composition. Rocks are mixtures of one or more minerals."),
    ("metal element conductive", "Metals are elements that conduct heat and electricity, are shiny and usually malleable. Iron, copper and aluminium are widely used structural metals."),
    ("alloy mixture of metals", "An alloy mixes a metal with other elements to improve properties. Steel is iron and carbon, and brass is copper and zinc."),
    ("polymer long chain molecule", "A polymer is a very long molecule made of repeated units. Plastics, DNA and proteins are natural or synthetic polymers."),
    ("philosopher love of wisdom", "A philosopher loves wisdom and asks big questions about life, knowledge and values. Socrates, Confucius and Descartes are famous philosophers."),
    ("ecology study of organisms", "Ecology is the branch of biology studying how organisms interact with each other and their environment, from populations to the whole biosphere."),
    ("metabolism body chemistry", "Metabolism is the set of chemical reactions keeping cells alive, converting food into energy and building or breaking down molecules."),
]

# ── web research: facts researched from public web sources (2026) and baked
#    in at build time, because the runtime sandbox has no general internet ───
_WEB_RESEARCH: list[tuple[str, str]] = [
    ("world population 2026 countries ranking", "World population reached about 8.2 billion in 2026. India is the most populous country with roughly 1.48 billion people, ahead of China at about 1.41 billion, the United States at 349 million and Indonesia at 288 million. Together India and China hold over a third of humanity."),
    ("most populous countries top ten", "The ten most populous countries in 2026 are India, China, the United States, Indonesia, Pakistan, Nigeria, Brazil, Bangladesh, Russia and Ethiopia. Pakistan has about 259 million people and Ethiopia about 139 million, and Sub-Saharan Africa is the fastest growing region."),
    ("world population growth peak", "The world's population grows by roughly seventy to eighty million people a year and is projected to peak around ten point three billion in the twenty-eighties before leveling off or declining, mainly because fertility rates have fallen across most of the world."),
    ("tallest building burj khalifa", "The Burj Khalifa in Dubai is the world's tallest building at 828 metres with 163 floors, completed in 2010. It leads by nearly 150 metres over Merdeka 118 in Kuala Lumpur at 679 metres and the Shanghai Tower at 632 metres."),
    ("skyscraper rankings tallest towers", "Among the world's tallest buildings are the Burj Khalifa at 828 metres, Merdeka 118 at 679 metres, Shanghai Tower at 632 metres, the Makkah Royal Clock Tower at 601 metres and One World Trade Center in New York at 541 metres. The Jeddah Tower, planned at over one thousand metres, may one day take the record."),
    ("most spoken languages world english", "English is the most spoken language in the world with about 1.49 billion total speakers, but Mandarin Chinese has the most native speakers at roughly 988 million. Hindi has about 611 million speakers and Spanish about 561 million, followed by Arabic and French."),
    ("native speakers mandarin spanish", "By native speakers alone the ranking is Mandarin Chinese at about 988 million, Spanish at 487 million, English at 372 million, Hindi at 347 million and Portuguese at about 252 million. About three quarters of English speakers learned it as a second language."),
    ("highest grossing film avatar box office", "Avatar, released in 2009, is the highest grossing film of all time with about 2.92 billion dollars worldwide. Avengers Endgame follows at about 2.80 billion, then Avatar The Way of Water at 2.32 billion and Titanic at 2.26 billion."),
    ("box office records movies", "The top films at the worldwide box office include Avatar, Avengers Endgame, Avatar The Way of Water, Titanic, Star Wars The Force Awakens and Avengers Infinity War, each earning around two billion dollars or more. James Cameron directed three of the biggest."),
    ("largest city tokyo jakarta delhi", "Tokyo is the largest city by metropolitan population with about 37 million people, followed by Delhi at about 33 million and Shanghai at about 30 million. Counting the whole built-up urban area, Jakarta in Indonesia leads with over 40 million people."),
    ("biggest cities population dhaka sao paulo", "Other giant cities include Dhaka, Sao Paulo, Mexico City, Cairo, Mumbai and Beijing, each home to twenty million people or more. New York City's metro area of about nineteen million is the largest in the Americas."),
    ("largest economy united states gdp", "The United States has the largest nominal economy in 2026 at about 32.6 trillion dollars, followed by China at about 21.5 trillion, Germany near 5.5 trillion, Japan and the United Kingdom around 4.3 trillion and India near 4.1 trillion."),
    ("gdp ranking ppp china india", "By purchasing power parity China is the largest economy at about 44 trillion dollars, ahead of the United States at 32 trillion and India at about 19 trillion. World gross domestic product is around 125 trillion dollars in 2026."),
    ("fastest growing major economy india", "India is among the fastest growing major economies, expanding at roughly six percent a year in 2026, while Japan's economy has been growing slowly and China's growth has moderated to around five percent."),
    ("record temperature hottest coldest earth", "The hottest reliably measured air temperature on Earth was fifty six point seven degrees Celsius at Furnace Creek Ranch in Death Valley in 1913, while the coldest was minus eighty nine point two degrees at Vostok Station in Antarctica in 1983."),
    ("deep ocean deepest point mariana", "The deepest point in the ocean is the Challenger Deep in the Mariana Trench, about ten thousand nine hundred and thirty five metres below the surface of the western Pacific. Only a handful of people have ever visited it in person."),
    ("olympic games facts largest sport event", "The modern Olympic Games are held every four years, alternating summer and winter editions. Paris hosted the Summer Games in 2024 and Los Angeles will host in 2028, and the Games trace back to ancient Greece where they were held at Olympia."),
    ("longest river nile amazon measured", "The Nile in Africa and the Amazon in South America compete for the title of longest river at about six thousand six hundred and fifty kilometres each. The Amazon carries far more water — around a fifth of all river discharge reaching the sea."),
    ("solar system facts eight planets", "The solar system has eight planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus and Neptune. Jupiter is the largest and Neptune the farthest of the eight, and dwarf planets like Pluto orbit beyond Neptune in the Kuiper belt."),
    ("largest desert sahara area", "The Sahara is the largest hot desert, covering about nine point two million square kilometers of northern Africa — roughly the size of the United States. Antarctica is the largest desert of any kind when dry areas are counted."),
    ("internet users world digital population", "More than five billion people use the internet, about two thirds of humanity. Asia has the most users, and mobile phones are the main way people in developing regions go online."),
    ("most valuable companies world", "The world's most valuable companies include Apple, Microsoft, Nvidia, Alphabet and Amazon, each worth trillions of dollars. Their value comes from software, chips, cloud computing, advertising and devices sold worldwide."),
    ("best selling video game minecraft", "Minecraft is the best selling video game of all time with over three hundred million copies sold. Tetris, Grand Theft Auto V and Wii Sports also rank among the top sellers in gaming history."),
    ("largest island greenland area", "Greenland is the world's largest island at about two point one million square kilometers, mostly covered by an ice sheet. Australia is larger but is classified as a continent, not an island."),
    ("longest bridge danyang kunshan china", "The Danyang Kunshan Grand Bridge in China is the longest bridge in the world at about one hundred and sixty five kilometres, carrying a high speed railway across the Yangtze River delta."),
    ("most visited country france tourism", "France is the most visited country in the world with around ninety million international tourists a year, followed by Spain and the United States. Paris alone attracts tens of millions of visitors annually."),
    ("highest mountain above sea level everest", "Mount Everest is the highest mountain above sea level at eight thousand eight hundred and forty nine metres. K2 is second at eight thousand six hundred and eleven metres, and the deepest ocean trench is deeper than Everest is tall."),
]

for _kw, _txt in _WEB_RESEARCH:
    PASSAGES.append((_kw, _txt))

# ── extend the passage pool ──────────────────────────────────────────────────
for _kw, _txt in (_PEOPLE + _ANIMALS + _INVENTIONS + _ARTS + _SPORTS +
                  _FOOD + _LANGUAGE + _SPACE + _HEALTH + _FINANCE + _GLOSSARY):
    PASSAGES.append((_kw, _txt))
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
        "what is {topic} ?",
        "who is associated with {topic} ?",
        "tell me more about {topic}",
        "why does {topic} matter ?",
        "give me information about {topic}",
        "define {topic}",
        "which {topic} is the most interesting ?",
        "do you know about {topic} ?",
        "describe {topic}",
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
