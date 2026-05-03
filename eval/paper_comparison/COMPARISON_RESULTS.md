# Paper Comparison Results

_Generated: 2026-04-27 23:38:06_


## Data Inventory

- BIRD metadata: 1533 questions

- Total per-query records: 14273

- Systems loaded:

  - Blind / Llama 4 Maverick: 1584 total (1315 BIRD)

  - Blind / Qwen 3 80B: 1584 total (1315 BIRD)

  - Blind / Sonnet 4.6: 1584 total (1315 BIRD)

  - CHESS: 555 total (555 BIRD)

  - DAIL-SQL: 1315 total (1315 BIRD)

  - DIN-SQL: 1315 total (1315 BIRD)

  - Pipeline V1 / Sonnet 4.6: 1584 total (1315 BIRD)

  - Pipeline V2 / Llama 4 Maverick: 1584 total (1315 BIRD)

  - Pipeline V2 / Qwen 3 80B: 1584 total (1315 BIRD)

  - Pipeline V2 / Sonnet 4.6: 1584 total (1315 BIRD)


---


## Section 1: Headline Comparison Table

All systems on BIRD queries. Pipeline and Blind may include Weaviate Gorilla queries; SOTA systems run on BIRD only.


| System                         | N (BIRD) | Strict/EX % | Relaxed % | Avg Input Tok/q | Avg Latency/q (s) | Avg Calls/q |
|--------------------------------|---------:|------------:|----------:|----------------:|------------------:|------------:|
| Pipeline V2 / Llama 4 Maverick | 1315     | 56.5%       | 92.4%     | 6,064           | 1.7               | 1           |
| Pipeline V2 / Qwen 3 80B       | 1315     | 58.8%       | 92.3%     | 5,746           | 2.1               | 1           |
| Pipeline V2 / Sonnet 4.6       | 1315     | 57.5%       | 93.1%     | 6,957           | 3.9               | 1           |
| Pipeline V1 / Sonnet 4.6       | 1315     | 55.7%       | 91.1%     | 4,338           | 3.5               | 1           |
| Blind / Llama 4 Maverick       | 1315     | 54.7%       | 92.1%     | 28,629          | 1.7               | 1           |
| Blind / Qwen 3 80B             | 1315     | 53.6%       | 91.8%     | 28,344          | 1.4               | 1           |
| Blind / Sonnet 4.6             | 1315     | 54.8%       | 93.5%     | 31,522          | 4.7               | 1           |
| CHESS                          | 555      | 62.7%       | 62.7%     | 196,945         | 30.1              | ~47         |
| DAIL-SQL                       | 1315     | 62.1%       | 62.1%     | 3,099           | 2.5               | 1           |
| DIN-SQL                        | 1315     | 63.5%       | 63.5%     | 23,409          | 16.8              | 4           |



---


## Section 2: V1 vs V2 Schema Comparison

Per-model comparison of pipeline V1 vs V2 tool schemas on BIRD queries.


| Model            | Schema | N         | Strict % | Relaxed % | JOIN Strict % | Single Strict % | Avg Input Tok |
|------------------|--------|----------:|---------:|----------:|--------------:|----------------:|--------------:|
| Llama 4 Maverick | V1     | [MISSING] | -        | -         | -             | -               | -             |
| Llama 4 Maverick | V2     | 1315      | 56.5%    | 92.4%     | 44.9%         | 89.9%           | 6,064         |
| Qwen 3 80B       | V1     | [MISSING] | -        | -         | -             | -               | -             |
| Qwen 3 80B       | V2     | 1315      | 58.8%    | 92.3%     | 47.4%         | 91.7%           | 5,746         |
| Sonnet 4.6       | V1     | 1315      | 55.7%    | 91.1%     | 44.6%         | 87.9%           | 4,338         |
| Sonnet 4.6       | V2     | 1315      | 57.5%    | 93.1%     | 45.8%         | 91.4%           | 6,957         |


**Deltas (V2 - V1):**

- **Llama 4 Maverick**: [MISSING one schema]

- **Qwen 3 80B**: [MISSING one schema]

- **Sonnet 4.6**: Strict +1.7pp, Relaxed +2.0pp



---


## Section 3: Pipeline Lift Over Blind Baseline

Per-model comparison: pipeline (best schema) vs blind baseline. Shows the value of the discovery phase.


| Model            | System        | N    | Strict % | Relaxed % | Avg Input Tok | Token Savings |
|------------------|---------------|-----:|---------:|----------:|--------------:|--------------:|
| Llama 4 Maverick | Pipeline (V2) | 1315 | 56.5%    | 92.4%     | 6,064         | -             |
| Llama 4 Maverick | Blind         | 1315 | 54.7%    | 92.1%     | 28,629        | 78.8%         |
| Qwen 3 80B       | Pipeline (V2) | 1315 | 58.8%    | 92.3%     | 5,746         | -             |
| Qwen 3 80B       | Blind         | 1315 | 53.6%    | 91.8%     | 28,344        | 79.7%         |
| Sonnet 4.6       | Pipeline (V2) | 1315 | 57.5%    | 93.1%     | 6,957         | -             |
| Sonnet 4.6       | Blind         | 1315 | 54.8%    | 93.5%     | 31,522        | 77.9%         |



---


## Section 4: Per-Database Breakdown (BIRD)

Strict accuracy (%) by database across all systems.


| Database      | N   | PV2 / Ll  | PV2 / Qw  | PV2 / Son | PV1 / Son | B/Ll      | B/Qw      | B/Son     | CHESS     | DAIL-SQL  | DIN-SQL   |
|---------------|----:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|
| CA Schools    | 77  | 55.8%     | 55.8%     | 61.0%     | 64.9%     | 66.2%     | 70.1%     | 66.2%     | 70.1%     | 63.6%     | 64.9%     |
| Card Games    | 163 | 84.0%     | 79.8%     | 85.3%     | 84.0%     | 68.1%     | 66.9%     | 65.6%     | 52.8%     | 46.6%     | 48.5%     |
| Codebase      | 159 | 57.2%     | 59.7%     | 56.6%     | 59.1%     | 58.5%     | 55.3%     | 55.3%     | -         | 64.2%     | 67.3%     |
| Debit Card    | 55  | 40.0%     | 49.1%     | 47.3%     | 50.9%     | 52.7%     | 52.7%     | 50.9%     | 72.7%     | 72.7%     | 70.9%     |
| Euro Football | 111 | 33.3%     | 33.3%     | 35.1%     | 31.5%     | 31.5%     | 35.1%     | 29.7%     | 73.0%     | 67.6%     | 64.0%     |
| Financial     | 91  | 36.3%     | 38.5%     | 38.5%     | 37.4%     | 39.6%     | 41.8%     | 40.7%     | -         | 62.6%     | 65.9%     |
| Formula 1     | 149 | 59.1%     | 57.0%     | 58.4%     | 56.4%     | 54.4%     | 49.7%     | 47.7%     | 58.4%     | 54.4%     | 57.0%     |
| Student Club  | 136 | 55.1%     | 53.7%     | 51.5%     | 50.7%     | 52.9%     | 47.8%     | 55.1%     | -         | 77.2%     | 77.9%     |
| Superhero     | 111 | 80.2%     | 84.7%     | 82.0%     | 73.0%     | 77.5%     | 75.7%     | 86.5%     | -         | 87.4%     | 84.7%     |
| Thrombosis    | 139 | 27.3%     | 48.9%     | 26.6%     | 19.4%     | 35.3%     | 33.1%     | 31.7%     | -         | 45.3%     | 47.5%     |
| Toxicology    | 124 | 72.6%     | 69.4%     | 76.6%     | 75.8%     | 61.3%     | 63.7%     | 72.6%     | -         | 57.3%     | 62.9%     |
| **TOTAL**     |     | **56.5%** | **58.8%** | **57.5%** | **55.7%** | **54.7%** | **53.6%** | **54.8%** | **62.7%** | **62.1%** | **63.5%** |


**Per-system best and worst databases:**

- **Pipeline V2 / Llama 4 Maverick**: Best = Card Games (84.0%), Worst = Thrombosis (27.3%)

- **Pipeline V2 / Qwen 3 80B**: Best = Superhero (84.7%), Worst = Euro Football (33.3%)

- **Pipeline V2 / Sonnet 4.6**: Best = Card Games (85.3%), Worst = Thrombosis (26.6%)

- **Pipeline V1 / Sonnet 4.6**: Best = Card Games (84.0%), Worst = Thrombosis (19.4%)

- **Blind / Llama 4 Maverick**: Best = Superhero (77.5%), Worst = Euro Football (31.5%)

- **Blind / Qwen 3 80B**: Best = Superhero (75.7%), Worst = Thrombosis (33.1%)

- **Blind / Sonnet 4.6**: Best = Superhero (86.5%), Worst = Euro Football (29.7%)

- **CHESS**: Best = Euro Football (73.0%), Worst = Card Games (52.8%)

- **DAIL-SQL**: Best = Superhero (87.4%), Worst = Thrombosis (45.3%)

- **DIN-SQL**: Best = Superhero (84.7%), Worst = Thrombosis (47.5%)



---


## Section 5: Per-Collection Accuracy (Pipeline Only)

Grouped by database. Shows accuracy for each collection across pipeline models.


| Database   | Collection                          | N   | Lla V2 | Qwe V2 | Son V1 | Son V2 |
|------------|-------------------------------------|----:|-------:|-------:|-------:|-------:|
| CA Schools | CaliforniaSchoolsFrpm               | 26  | 38.5%  | 42.3%  | 34.6%  | 38.5%  |
| CA Schools | CaliforniaSchoolsSatscores          | 23  | 43.5%  | 43.5%  | 73.9%  | 65.2%  |
| CA Schools | CaliforniaSchoolsSchools            | 28  | 82.1%  | 78.6%  | 85.7%  | 78.6%  |
| Card Games | CardGamesCards                      | 112 | 91.1%  | 87.5%  | 92.0%  | 93.8%  |
| Card Games | CardGamesForeignData                | 6   | 66.7%  | 33.3%  | 50.0%  | 50.0%  |
| Card Games | CardGamesLegalities                 | 2   | 100.0% | 100.0% | 100.0% | 100.0% |
| Card Games | CardGamesSetTranslations            | 8   | 75.0%  | 75.0%  | 62.5%  | 62.5%  |
| Card Games | CardGamesSets                       | 35  | 65.7%  | 62.9%  | 68.6%  | 68.6%  |
| Codebase   | CodebaseCommunityBadges             | 15  | 80.0%  | 86.7%  | 60.0%  | 80.0%  |
| Codebase   | CodebaseCommunityComments           | 19  | 36.8%  | 47.4%  | 42.1%  | 36.8%  |
| Codebase   | CodebaseCommunityPosthistory        | 5   | 40.0%  | 20.0%  | 20.0%  | 40.0%  |
| Codebase   | CodebaseCommunityPostlinks          | 2   | 50.0%  | 50.0%  | 50.0%  | 50.0%  |
| Codebase   | CodebaseCommunityPosts              | 39  | 79.5%  | 79.5%  | 69.2%  | 74.4%  |
| Codebase   | CodebaseCommunityTags               | 6   | 50.0%  | 66.7%  | 83.3%  | 66.7%  |
| Codebase   | CodebaseCommunityUsers              | 65  | 43.1%  | 44.6%  | 56.9%  | 43.1%  |
| Codebase   | CodebaseCommunityVotes              | 8   | 87.5%  | 87.5%  | 75.0%  | 87.5%  |
| Debit Card | DebitCardSpecializingCustomers      | 17  | 29.4%  | 41.2%  | 29.4%  | 17.6%  |
| Debit Card | DebitCardSpecializingGasstations    | 4   | 75.0%  | 75.0%  | 100.0% | 100.0% |
| Debit Card | DebitCardSpecializingTransactions1k | 27  | 29.6%  | 44.4%  | 51.9%  | 48.1%  |
| Debit Card | DebitCardSpecializingYearmonth      | 7   | 85.7%  | 71.4%  | 71.4%  | 85.7%  |
| Euro Footb | EuropeanFootball2Country            | 7   | 14.3%  | 0.0%   | 14.3%  | 14.3%  |
| Euro Footb | EuropeanFootball2League             | 12  | 0.0%   | 0.0%   | 0.0%   | 0.0%   |
| Euro Footb | EuropeanFootball2Match              | 3   | 100.0% | 100.0% | 100.0% | 100.0% |
| Euro Footb | EuropeanFootball2Player             | 55  | 29.1%  | 29.1%  | 23.6%  | 29.1%  |
| Euro Footb | EuropeanFootball2PlayerAttributes   | 14  | 92.9%  | 100.0% | 92.9%  | 100.0% |
| Euro Footb | EuropeanFootball2Team               | 18  | 11.1%  | 11.1%  | 16.7%  | 16.7%  |
| Euro Footb | EuropeanFootball2TeamAttributes     | 2   | 100.0% | 100.0% | 100.0% | 100.0% |
| Financial  | FinancialAccount                    | 25  | 24.0%  | 28.0%  | 20.0%  | 28.0%  |
| Financial  | FinancialCard                       | 5   | 60.0%  | 60.0%  | 60.0%  | 60.0%  |
| Financial  | FinancialClient                     | 20  | 30.0%  | 30.0%  | 25.0%  | 15.0%  |
| Financial  | FinancialDisp                       | 5   | 0.0%   | 0.0%   | 20.0%  | 0.0%   |
| Financial  | FinancialDistrict                   | 21  | 47.6%  | 52.4%  | 47.6%  | 57.1%  |
| Financial  | FinancialLoan                       | 8   | 50.0%  | 62.5%  | 87.5%  | 87.5%  |
| Financial  | FinancialTrans                      | 7   | 57.1%  | 42.9%  | 42.9%  | 42.9%  |
| Formula 1  | Formula1Circuits                    | 33  | 69.7%  | 66.7%  | 60.6%  | 63.6%  |
| Formula 1  | Formula1Constructorresults          | 3   | 33.3%  | 33.3%  | 33.3%  | 33.3%  |
| Formula 1  | Formula1Constructors                | 1   | 0.0%   | 0.0%   | 0.0%   | 0.0%   |
| Formula 1  | Formula1Constructorstandings        | 3   | 100.0% | 33.3%  | 66.7%  | 33.3%  |
| Formula 1  | Formula1Drivers                     | 27  | 51.9%  | 51.9%  | 48.1%  | 48.1%  |
| Formula 1  | Formula1Driverstandings             | 3   | 33.3%  | 66.7%  | 33.3%  | 66.7%  |
| Formula 1  | Formula1Laptimes                    | 12  | 83.3%  | 83.3%  | 75.0%  | 75.0%  |
| Formula 1  | Formula1Pitstops                    | 5   | 80.0%  | 80.0%  | 60.0%  | 80.0%  |
| Formula 1  | Formula1Qualifying                  | 8   | 75.0%  | 87.5%  | 75.0%  | 75.0%  |
| Formula 1  | Formula1Races                       | 30  | 40.0%  | 40.0%  | 53.3%  | 50.0%  |
| Formula 1  | Formula1Results                     | 24  | 58.3%  | 50.0%  | 54.2%  | 62.5%  |
| Student Cl | StudentClubAttendance               | 2   | 0.0%   | 0.0%   | 0.0%   | 0.0%   |
| Student Cl | StudentClubBudget                   | 17  | 82.4%  | 82.4%  | 82.4%  | 88.2%  |
| Student Cl | StudentClubEvent                    | 36  | 61.1%  | 44.4%  | 47.2%  | 50.0%  |
| Student Cl | StudentClubExpense                  | 12  | 83.3%  | 75.0%  | 83.3%  | 83.3%  |
| Student Cl | StudentClubIncome                   | 4   | 75.0%  | 75.0%  | 75.0%  | 75.0%  |
| Student Cl | StudentClubMajor                    | 11  | 72.7%  | 45.5%  | 63.6%  | 63.6%  |
| Student Cl | StudentClubMember                   | 51  | 29.4%  | 45.1%  | 29.4%  | 27.5%  |
| Student Cl | StudentClubZipCode                  | 3   | 100.0% | 100.0% | 100.0% | 100.0% |
| Superhero  | SuperheroHeroAttribute              | 4   | 100.0% | 100.0% | 100.0% | 100.0% |
| Superhero  | SuperheroHeroPower                  | 7   | 28.6%  | 71.4%  | 42.9%  | 100.0% |
| Superhero  | SuperheroPublisher                  | 1   | 100.0% | 100.0% | 100.0% | 100.0% |
| Superhero  | SuperheroSuperhero                  | 98  | 82.7%  | 84.7%  | 73.5%  | 79.6%  |
| Superhero  | SuperheroSuperpower                 | 1   | 100.0% | 100.0% | 100.0% | 100.0% |
| Thrombosis | ThrombosisPredictionExamination     | 12  | 66.7%  | 66.7%  | 58.3%  | 66.7%  |
| Thrombosis | ThrombosisPredictionLaboratory      | 10  | 70.0%  | 80.0%  | 70.0%  | 80.0%  |
| Thrombosis | ThrombosisPredictionPatient         | 117 | 19.7%  | 44.4%  | 11.1%  | 17.9%  |
| Toxicology | ToxicologyAtom                      | 66  | 78.8%  | 66.7%  | 81.8%  | 84.8%  |
| Toxicology | ToxicologyBond                      | 34  | 64.7%  | 76.5%  | 76.5%  | 67.6%  |
| Toxicology | ToxicologyConnected                 | 7   | 28.6%  | 28.6%  | 28.6%  | 28.6%  |
| Toxicology | ToxicologyMolecule                  | 17  | 82.4%  | 82.4%  | 70.6%  | 82.4%  |


**Hardest collections (lowest average strict accuracy, N >= 5):**

- EuropeanFootball2League (Euro Football): 0.0% avg strict

- FinancialDisp (Financial): 5.0% avg strict

- EuropeanFootball2Country (Euro Football): 10.7% avg strict

- EuropeanFootball2Team (Euro Football): 13.9% avg strict

- ThrombosisPredictionPatient (Thrombosis): 23.3% avg strict

- FinancialClient (Financial): 25.0% avg strict

- FinancialAccount (Financial): 25.0% avg strict

- EuropeanFootball2Player (Euro Football): 27.7% avg strict

- ToxicologyConnected (Toxicology): 28.6% avg strict

- DebitCardSpecializingCustomers (Debit Card): 29.4% avg strict

- CodebaseCommunityPosthistory (Codebase): 30.0% avg strict

- StudentClubMember (Student Club): 32.8% avg strict

- CaliforniaSchoolsFrpm (CA Schools): 38.5% avg strict

- CodebaseCommunityComments (Codebase): 40.8% avg strict

- DebitCardSpecializingTransactions1k (Debit Card): 43.5% avg strict


**Top confusion pairs (predicted -> expected):**

- ThrombosisPredictionLaboratory -> ThrombosisPredictionPatient: 286x

- EuropeanFootball2PlayerAttributes -> EuropeanFootball2Player: 140x

- CodebaseCommunityPosts -> CodebaseCommunityUsers: 70x

- ThrombosisPredictionExamination -> ThrombosisPredictionPatient: 62x

- CaliforniaSchoolsSchools -> CaliforniaSchoolsFrpm: 58x

- EuropeanFootball2TeamAttributes -> EuropeanFootball2Team: 54x

- Formula1Results -> Formula1Races: 46x

- DebitCardSpecializingYearmonth -> DebitCardSpecializingCustomers: 46x

- Formula1Races -> Formula1Circuits: 45x

- EuropeanFootball2Match -> EuropeanFootball2League: 44x

- CodebaseCommunityBadges -> CodebaseCommunityUsers: 42x

- StudentClubMajor -> StudentClubMember: 40x

- StudentClubBudget -> StudentClubEvent: 36x

- CardGamesSetTranslations -> CardGamesSets: 35x

- CaliforniaSchoolsSchools -> CaliforniaSchoolsSatscores: 34x

- ToxicologyMolecule -> ToxicologyAtom: 31x

- SuperheroHeroAttribute -> SuperheroSuperhero: 30x

- Formula1Results -> Formula1Drivers: 29x

- StudentClubZipCode -> StudentClubMember: 25x

- StudentClubExpense -> StudentClubMember: 24x



---


## Section 6: Accuracy by Query Difficulty

BIRD difficulty levels: simple, moderate, challenging.


| System                         | Simple        | Moderate      | Challenging   | Overall |
|--------------------------------|--------------:|--------------:|--------------:|--------:|
| Pipeline V2 / Llama 4 Maverick | 62.5% (n=785) | 45.2% (n=403) | 55.1% (n=127) | 56.5%   |
| Pipeline V2 / Qwen 3 80B       | 64.8% (n=785) | 49.4% (n=403) | 51.2% (n=127) | 58.8%   |
| Pipeline V2 / Sonnet 4.6       | 65.0% (n=785) | 43.9% (n=403) | 54.3% (n=127) | 57.5%   |
| Pipeline V1 / Sonnet 4.6       | 63.7% (n=785) | 41.7% (n=403) | 51.2% (n=127) | 55.7%   |
| Blind / Llama 4 Maverick       | 60.9% (n=785) | 43.2% (n=403) | 52.8% (n=127) | 54.7%   |
| Blind / Qwen 3 80B             | 60.0% (n=785) | 42.4% (n=403) | 49.6% (n=127) | 53.6%   |
| Blind / Sonnet 4.6             | 61.3% (n=785) | 42.4% (n=403) | 53.5% (n=127) | 54.8%   |
| CHESS                          | 62.7% (n=346) | 62.8% (n=164) | 62.2% (n=45)  | 62.7%   |
| DAIL-SQL                       | 66.6% (n=785) | 56.6% (n=403) | 51.2% (n=127) | 62.1%   |
| DIN-SQL                        | 68.4% (n=785) | 56.8% (n=403) | 54.3% (n=127) | 63.5%   |


**Difficulty-level advantage analysis:**

- **Simple**: Pipeline best 65.0% (n=785) vs SOTA best 68.4% (n=785) (-3.4pp)

- **Moderate**: Pipeline best 49.4% (n=403) vs SOTA best 62.8% (n=164) (-13.4pp)

- **Challenging**: Pipeline best 55.1% (n=127) vs SOTA best 62.2% (n=45) (-7.1pp)



---


## Section 7: Accuracy by SQL Feature

Queries tagged by features detected via regex on gold SQL.
A query can have multiple features.


| Feature               | N (uniq q) | PV2 / Ll | PV2 / Qw | PV2 / Son | PV1 / Son | B/Ll  | B/Qw  | B/Son | CHESS | DAIL-SQL | DIN-SQL |
|-----------------------|-----------:|---------:|---------:|----------:|----------:|------:|------:|------:|------:|---------:|--------:|
| JOIN                  | 976        | 44.9%    | 47.4%    | 45.8%     | 44.6%     | 40.9% | 39.9% | 40.8% | 60.6% | 61.0%    | 62.1%   |
| WHERE+AND             | 554        | 52.2%    | 57.0%    | 52.9%     | 50.5%     | 52.2% | 50.2% | 51.6% | 67.2% | 63.5%    | 64.4%   |
| COUNT                 | 433        | 60.5%    | 62.1%    | 60.3%     | 58.2%     | 59.4% | 57.0% | 58.4% | 71.9% | 64.9%    | 66.5%   |
| ORDER BY              | 269        | 52.4%    | 53.5%    | 54.6%     | 53.5%     | 49.8% | 52.4% | 51.3% | 62.7% | 56.1%    | 55.8%   |
| LIMIT                 | 258        | 51.9%    | 53.5%    | 54.3%     | 52.7%     | 49.2% | 51.9% | 50.4% | 63.3% | 57.8%    | 56.2%   |
| AGG (SUM/AVG/MAX/MIN) | 223        | 58.3%    | 59.2%    | 57.8%     | 53.8%     | 52.5% | 51.1% | 52.9% | 66.1% | 60.5%    | 62.3%   |
| DISTINCT              | 215        | 52.1%    | 56.7%    | 54.0%     | 49.8%     | 41.4% | 39.1% | 44.7% | 50.0% | 42.8%    | 42.3%   |
| Subquery              | 111        | 64.9%    | 62.2%    | 64.0%     | 59.5%     | 65.8% | 65.8% | 67.6% | 54.7% | 49.5%    | 47.7%   |
| GROUP BY              | 107        | 59.8%    | 58.9%    | 57.0%     | 54.2%     | 45.8% | 51.4% | 47.7% | 61.4% | 51.4%    | 56.1%   |
| CASE WHEN             | 94         | 72.3%    | 72.3%    | 71.3%     | 68.1%     | 72.3% | 63.8% | 71.3% | 48.6% | 53.2%    | 51.1%   |
| BETWEEN               | 64         | 48.4%    | 57.8%    | 51.6%     | 42.2%     | 51.6% | 59.4% | 57.8% | 77.3% | 62.5%    | 67.2%   |
| WHERE+OR              | 35         | 62.9%    | 62.9%    | 57.1%     | 57.1%     | 65.7% | 62.9% | 57.1% | 75.0% | 51.4%    | 57.1%   |


**Feature interactions (pipeline V2 / best model, BIRD only):**

- JOIN only: 48.3% (n=735)

- JOIN + ORDER BY: 41.3% (n=201)

- JOIN + Subquery: 56.9% (n=58)

- No JOIN: 91.7% (n=338)

- COUNT + GROUP BY: 60.3% (n=68)

- WHERE+AND (no JOIN): 92.1% (n=114)



---


## Section 8: Pattern Analysis

### Top 20 Confusion Pairs (all pipeline runs)

Counts how often collection X was predicted when Y was expected.


| Predicted                         | Expected                       | Count |
|-----------------------------------|--------------------------------|------:|
| ThrombosisPredictionLaboratory    | ThrombosisPredictionPatient    | 286   |
| EuropeanFootball2PlayerAttributes | EuropeanFootball2Player        | 140   |
| CodebaseCommunityPosts            | CodebaseCommunityUsers         | 70    |
| ThrombosisPredictionExamination   | ThrombosisPredictionPatient    | 62    |
| CaliforniaSchoolsSchools          | CaliforniaSchoolsFrpm          | 58    |
| EuropeanFootball2TeamAttributes   | EuropeanFootball2Team          | 54    |
| Formula1Results                   | Formula1Races                  | 46    |
| DebitCardSpecializingYearmonth    | DebitCardSpecializingCustomers | 46    |
| Formula1Races                     | Formula1Circuits               | 45    |
| EuropeanFootball2Match            | EuropeanFootball2League        | 44    |
| CodebaseCommunityBadges           | CodebaseCommunityUsers         | 42    |
| StudentClubMajor                  | StudentClubMember              | 40    |
| StudentClubBudget                 | StudentClubEvent               | 36    |
| CardGamesSetTranslations          | CardGamesSets                  | 35    |
| CaliforniaSchoolsSchools          | CaliforniaSchoolsSatscores     | 34    |
| ToxicologyMolecule                | ToxicologyAtom                 | 31    |
| SuperheroHeroAttribute            | SuperheroSuperhero             | 30    |
| Formula1Results                   | Formula1Drivers                | 29    |
| StudentClubZipCode                | StudentClubMember              | 25    |
| StudentClubExpense                | StudentClubMember              | 24    |


### Universal Hard Cases

Queries where ALL systems with data fail (strict=False for every system).


Found **95** queries where all 10 systems fail.


1. [thrombosis_prediction] [simple] How many patients who were female got white blood cells that were below 3.5?

2. [california_schools] [challenging] What are the valid e-mail addresses of the administrator of the school located in the San Bernardino county, City of San Bernardino City Unified that opened between 1/1/2009 to 12/31/2010 whose school types are public Intermediate/Middle Schools and Unified Schools?

3. [european_football_2] [simple] What is the passing class of CLB team?

4. [codebase_community] [simple] Which post has the highest score? Please give its id and title's name.

5. [superhero] [simple] What is the race of the superhero with maximum attribute value?

6. [financial] [moderate] How many of the account holders in South Bohemia still do not own credit cards?

7. [financial] [moderate] How many accounts in North Bohemia has made a transaction with the partner's bank being AB?

8. [formula_1] [moderate] Which driver ranked the first in the Canadian Grand Prix in 2007? Please give his reference name.

9. [codebase_community] [challenging] Based on posts posted by Community, calculate the percentage of posts that use the R language.

10. [codebase_community] [moderate] State all the tags used by Mark Meckes in his posts that doesn't have comments.

11. [student_club] [moderate] Did Maya Mclean attend the 'Women's Soccer' event?

12. [thrombosis_prediction] [moderate] How many patients have a normal level of anti-ribonuclear protein and have been admitted to the hospital?

13. [financial] [moderate] How many male customers who were born between 1974 and 1976 have made a payment on their home in excess of $4000?

14. [card_games] [moderate] Find and list the names of sets which doesn't have Japanese translation but have Korean translation.

15. [financial] [moderate] Which district has highest active loan?

16. [thrombosis_prediction] [moderate] Among the patients whose total cholesterol is within the normal range, how many of them have a P pattern observed in the sheet of ANA examination?

17. [formula_1] [simple] Name all drivers in the 2010 Singapore Grand Prix order by their position stands.

18. [formula_1] [challenging] Please list the lap records for the circuits in Italy.

19. [formula_1] [simple] Which drivers born after 1975 have been ranked 2? Please give their forenames and surnames.

20. [financial] [simple] Which districts have transactions greater than USS$10,000 in 1997?


... and 75 more.


### Queries Where Pipeline Succeeds but SOTA Fails

**250** queries where at least one pipeline run succeeds but no SOTA system does.

1. [financial] [challenging] List out the account numbers of female clients who are oldest and has lowest average salary, calculate the gap between t

2. [student_club] [simple] Among the budgets for Advertising, list out top three which have the most budgeted amount?

3. [thrombosis_prediction] [moderate] How many male patients have a normal level of both albumin and total protein?

4. [formula_1] [simple] How many races were there in 2005? Name all the races in descending order.

5. [student_club] [simple] What is the ratio between students majored in finance and physics?

6. [card_games] [challenging] Which set is not available outside of the United States and has foil cards with Japanese writing on them? Please include

7. [european_football_2] [simple] What is the defensive work rate of the football player David Wilson
?

8. [card_games] [simple] To which artist does the card with the text "Das perfekte Gegenmittel zu einer dichten Formation" belong?

9. [formula_1] [simple] For the driver who had the Q2 time as 0:01:40 in the qualifying race No. 355, what is his nationality?

10. [formula_1] [simple] In which race did the fastest 1st lap time was recorded? Please indicate the time in milliseconds.

11. [toxicology] [moderate] What is the atom ID of double bonded carbon in TR012 molecule?

12. [card_games] [simple] Which are the cards that have incredibly powerful foils.

13. [toxicology] [moderate] Which molecule does the atom TR001_10 belong to? Please state whether this molecule is carcinogenic or not.

14. [card_games] [challenging] What is the annual average number of sets that were released between 1/1/2012 to 12/31/2015? Indicate the common languga

15. [card_games] [simple] Pick 3 cards with rarity of uncommon, list down name these cards according to ascending order of it's ruling date.


... and 235 more.


_Pipeline-unique successes by DB:_ 
Card Games (60), Formula 1 (34), Thrombosis (32), Toxicology (28), Codebase (28), Student Club (16), Financial (14), Euro Football (12), Superhero (11), CA Schools (9)



### Queries Where SOTA Succeeds but Pipeline Fails

**276** queries where at least one SOTA system succeeds but no pipeline run does.

1. [european_football_2] [simple] What is the average overall rating of the football player Aaron Doran?

2. [student_club] [simple] State what kind of expenses that Sacha Harrison incurred?

3. [european_football_2] [moderate] Give the name of the league with the highest matches of all time and how many matches were played in the said league.

4. [financial] [simple] For the female client who was born in 1976/1/29, which district did she opened her account?

5. [codebase_community] [simple] What is the date when the youngest user made his or her first post?

6. [codebase_community] [simple] What is the display name of the user who is the owner of the most valuable post?

7. [european_football_2] [moderate] Please list the names of the players whose volley score and dribbling score are over 70.

8. [european_football_2] [moderate] How many matches were held in the league Germany 1. Bundesliga
from August to October 2008?

9. [student_club] [simple] State the name of students from Georgetown, South Carolina.

10. [formula_1] [simple] Which website should I go to if I want to know more about Anthony Davidson?

11. [student_club] [moderate] Among the students majored in interior design, who have attended the Community Theater event?

12. [european_football_2] [moderate] What was the potiential for Francesco Parravicini on 2010/8/30?

13. [formula_1] [moderate] State the driver with the most points scored. Find his full name with that points.

14. [california_schools] [simple] What is the phone number of the school that has the highest average score in Math?

15. [financial] [simple] List out the id number of client who choose statement of issuance after transaction are Disponent?


... and 261 more.


_SOTA-unique successes by DB:_ 
Euro Football (58), Student Club (38), Formula 1 (33), Thrombosis (32), Financial (31), Codebase (26), CA Schools (17), Debit Card (14), Toxicology (12), Card Games (8), Superhero (7)




---


## Section 9: Latency Analysis

### Latency Distribution (seconds/query)


| System                         | N    | Mean | Median | P95  | P99  | Min  | Max   |
|--------------------------------|-----:|-----:|-------:|-----:|-----:|-----:|------:|
| Pipeline V2 / Llama 4 Maverick | 1584 | 1.7  | 1.2    | 4.7  | 6.5  | 0.4  | 10.9  |
| Pipeline V2 / Qwen 3 80B       | 1584 | 2.0  | 1.4    | 2.5  | 3.4  | 0.8  | 364.5 |
| Pipeline V2 / Sonnet 4.6       | 1584 | 3.7  | 3.6    | 5.6  | 7.2  | 1.7  | 31.1  |
| Pipeline V1 / Sonnet 4.6       | 1584 | 3.4  | 3.2    | 5.2  | 7.4  | 1.5  | 16.8  |
| Blind / Llama 4 Maverick       | 1584 | 1.7  | 1.3    | 3.2  | 3.8  | 0.5  | 9.8   |
| Blind / Qwen 3 80B             | 1584 | 1.4  | 1.3    | 2.2  | 3.2  | 1.2  | 13.9  |
| Blind / Sonnet 4.6             | 1584 | 4.6  | 4.4    | 7.3  | 10.1 | 2.6  | 21.8  |
| CHESS                          | 555  | 30.1 | 23.4   | 53.8 | 67.9 | 15.4 | 79.4  |
| DAIL-SQL                       | 1315 | 2.5  | 2.2    | 4.3  | 6.3  | 1.1  | 25.4  |
| DIN-SQL                        | 1315 | 16.8 | 17.0   | 22.7 | 27.6 | 7.2  | 38.3  |


### Latency by Database (mean seconds, pipeline only)


| Database      | PV2 / Ll | PV2 / Qw | PV2 / Son | PV1 / Son |
|---------------|---------:|---------:|----------:|----------:|
| CA Schools    | 1.4      | 1.7      | 4.4       | 3.3       |
| Card Games    | 1.6      | 1.5      | 3.7       | 3.3       |
| Codebase      | 1.6      | 1.3      | 3.4       | 3.2       |
| Debit Card    | 1.6      | 1.8      | 4.4       | 3.7       |
| Euro Football | 1.8      | 1.4      | 3.8       | 3.5       |
| Financial     | 1.9      | 1.6      | 4.3       | 3.8       |
| Formula 1     | 1.9      | 1.7      | 3.8       | 3.4       |
| Student Club  | 1.5      | 1.4      | 3.8       | 3.6       |
| Superhero     | 1.8      | 1.5      | 4.1       | 3.5       |
| Thrombosis    | 1.9      | 1.6      | 4.1       | 3.8       |
| Toxicology    | 1.8      | 7.4      | 3.7       | 3.6       |


### Latency by Difficulty (mean seconds)


| System                         | Simple | Moderate | Challenging |
|--------------------------------|-------:|---------:|------------:|
| Pipeline V2 / Llama 4 Maverick | 1.6    | 1.8      | 2.1         |
| Pipeline V2 / Qwen 3 80B       | 1.4    | 2.9      | 3.6         |
| Pipeline V2 / Sonnet 4.6       | 3.5    | 4.3      | 4.6         |
| Pipeline V1 / Sonnet 4.6       | 3.3    | 3.8      | 3.9         |
| Blind / Llama 4 Maverick       | 1.7    | 1.8      | 1.9         |
| Blind / Qwen 3 80B             | 1.4    | 1.5      | 1.4         |
| Blind / Sonnet 4.6             | 4.6    | 5.0      | 5.0         |
| CHESS                          | 28.7   | 31.8     | 35.3        |
| DAIL-SQL                       | 2.2    | 2.7      | 3.1         |
| DIN-SQL                        | 16.0   | 17.9     | 18.4        |



---


## Section 10: Token Efficiency Deep Dive

### Token Distribution (BIRD queries)


| System                         | N    | Mean In | Median In | P95 In  | Mean Out | Median Out | P95 Out | Mean Total |
|--------------------------------|-----:|--------:|----------:|--------:|---------:|-----------:|--------:|-----------:|
| Pipeline V2 / Llama 4 Maverick | 1315 | 6,064   | 5,908     | 7,294   | 195      | 197        | 340     | 6,260      |
| Pipeline V2 / Qwen 3 80B       | 1315 | 5,746   | 5,599     | 6,828   | 81       | 74         | 139     | 5,828      |
| Pipeline V2 / Sonnet 4.6       | 1315 | 6,957   | 6,881     | 7,871   | 291      | 296        | 476     | 7,248      |
| Pipeline V1 / Sonnet 4.6       | 1315 | 4,338   | 4,159     | 5,703   | 196      | 203        | 321     | 4,534      |
| Blind / Llama 4 Maverick       | 1315 | 28,629  | 28,629    | 28,643  | 82       | 65         | 190     | 28,711     |
| Blind / Qwen 3 80B             | 1315 | 28,344  | 28,343    | 28,360  | 32       | 32         | 35      | 28,376     |
| Blind / Sonnet 4.6             | 1315 | 31,522  | 31,522    | 31,536  | 186      | 197        | 289     | 31,708     |
| CHESS                          | 555  | 196,945 | 22,968    | 906,723 | 5,666    | 1,150      | 22,101  | 202,612    |
| DAIL-SQL                       | 1315 | 3,099   | 3,133     | 5,393   | 80       | 67         | 184     | 3,180      |
| DIN-SQL                        | 1315 | 23,409  | 23,816    | 32,383  | 800      | 797        | 1,138   | 24,210     |


### Tokens by Number of JOINs in Gold SQL


_Reference system: Pipeline V2 / Llama 4 Maverick_


| # JOINs | N   | Mean Input Tok | Strict % |
|---------|----:|---------------:|---------:|
| 0       | 338 | 6,000          | 89.9%    |
| 1       | 767 | 6,088          | 45.9%    |
| 2       | 177 | 6,073          | 42.4%    |
| 3+      | 33  | 6,104          | 36.4%    |


### Cost Comparison ($3/1M input, $15/1M output)


Estimated cost for N=1,000 queries at Bedrock Sonnet 4.6 pricing.


| System                         | Avg In Tok | Avg Out Tok | Cost/Query | Cost/1000q | Relative |
|--------------------------------|-----------:|------------:|-----------:|-----------:|---------:|
| Pipeline V2 / Llama 4 Maverick | 6,064      | 195         | $0.0211    | $21.13     | 2.0x     |
| Pipeline V2 / Qwen 3 80B       | 5,746      | 81          | $0.0185    | $18.47     | 1.8x     |
| Pipeline V2 / Sonnet 4.6       | 6,957      | 291         | $0.0252    | $25.24     | 2.4x     |
| Pipeline V1 / Sonnet 4.6       | 4,338      | 196         | $0.0160    | $15.97     | 1.5x     |
| Blind / Llama 4 Maverick       | 28,629     | 82          | $0.0871    | $87.12     | 8.3x     |
| Blind / Qwen 3 80B             | 28,344     | 32          | $0.0855    | $85.52     | 8.1x     |
| Blind / Sonnet 4.6             | 31,522     | 186         | $0.0974    | $97.36     | 9.3x     |
| CHESS                          | 196,945    | 5,666       | $0.6758    | $675.84    | 64.3x    |
| DAIL-SQL                       | 3,099      | 80          | $0.0105    | $10.50     | 1.0x     |
| DIN-SQL                        | 23,409     | 800         | $0.0822    | $82.24     | 7.8x     |


### Token Savings Summary


Comparing pipeline (progressive disclosure) to blind (all 143 schemas) and SOTA systems.


- **Llama 4 Maverick**: Pipeline 6064 tok/q vs Blind 28629 tok/q = **78.8% savings**

- **Qwen 3 80B**: Pipeline 5747 tok/q vs Blind 28344 tok/q = **79.7% savings**

- **Sonnet 4.6**: Pipeline 5648 tok/q vs Blind 31523 tok/q = **82.1% savings**



---

