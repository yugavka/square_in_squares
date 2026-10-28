# Источники импортированных раскладок

`alexchilton_22262_delta.csv.gz` — 105 раскладок (66 680 строк) из
[публичного ноутбука Fault-line DP + Basin Hopping - LB 22.2672](https://www.kaggle.com/code/alexchilton/fault-line-dp-basin-hopping-lb-22-2672)
(запуск 2026-10-28 16:29 UTC, публичный балл `22.262381844`), сторона которых
строго меньше, чем в обоих снимках ntumlnoob. Исходный `submission.csv`
скачан командой `kaggle kernels output alexchilton/fault-line-dp-basin-hopping-lb-22-2672`,
содержит 500 500 строк, прошёл локальный валидатор с баллом
`22.262381844045` и имеет SHA-256
`167e9c66d9e712ea5e0327d5495a6c7a4ca714eb6c922714108e96ab6b1b5a2d`.
Сжатый снимок сохраняет исходные десятичные строки; SHA-256 архива:
`28af16a528f381041fbcdf619d6b3eb74ba0b8af9d91f10e3766acca4ade3852`.
Ноутбук объединяет публичные портфели; [метки
происхождения](alexchilton_22262_provenance.csv) взяты из его
`merged_sources.csv`: 98 раскладок помечены ноутбуком «LB 22.2683156 Public
Packings + FQ Quench», остальные 7 — Neighbour Transfer + Squeeze, Deep EDA &
Analogies и C++ Seam DP.

`ntumlnoob_2264_delta.csv.gz` — 94 раскладки (67 697 строк) из
[публичного ноутбука 22.264 With rectangles/FTMC/MILP/FQ/cpu+gpu](https://www.kaggle.com/code/ntumlnoob/22-264-with-rectangles-ftmc-milp-fq-cpu-gpu?scriptVersionId=363062691),
сторона которых строго меньше, чем в снимке `ntumlnoob_2268_partial.csv.gz`.
Первая версия снимка содержала 73 раскладки, улучшавшие наш файл больше чем на
`10⁻⁹`, и теряла ещё 21 улучшение порядка `10⁻⁹`.
Исходный опубликованный `submission.csv` содержит 500 500 строк, прошёл
локальный валидатор и имеет SHA-256
`821b0389409faa5fb369be8f0f995e72b08a840926c619884f54b661a64b5355`.
Этот хеш также указан в самом ноутбуке. Сжатый снимок сохраняет исходные
десятичные координаты; SHA-256 архива:
`49b6626d7a61dfcb1bf197a2b7391645d8db1ac6542c26750904767dd13d5da4`.
Сам ноутбук описывает сборку прямоугольных фрагментов, FTMC/MILP/FQ и
объединение публичных портфелей. Его итоговый файл включает результаты
нескольких авторов; опубликованные
[`SOURCE_NOTICES.md`](ntumlnoob_2264_SOURCE_NOTICES.md) и
[метки происхождения для выбранных 94 размеров](ntumlnoob_2264_provenance.csv)
сохранены рядом с координатами. Из них 38 помечены CPU FTMC/FQ, 14 — GPU/FQ,
две — сборкой прямоугольников и FQ, остальные 40 — публичными донорами.

`ntumlnoob_2268_partial.csv.gz` — базовый снимок, также используемый по умолчанию,
[публичного ноутбука First 22.268681355 Public Code With Explanation](https://www.kaggle.com/code/ntumlnoob/first-22-268681355-public-code-with-explanation).
Он получен из опубликованного `submission.csv` с SHA-256
`30dded7f9b2ab419c87daceb6d72f53502edbe23238caea3301dac9110720a7b`.
Сохранены 547 раскладок (290 670 строк), которые улучшают обычную решётку;
остальные размеры генератор заполняет самостоятельно. Сжатый снимок и полный
CSV дают одинаковый локальный балл `22.268681355090` и прошли локальный
валидатор. Ноутбук включает координаты из нескольких публичных портфелей и
собственных улучшений автора; подробные ссылки на исходные источники находятся
в его `SOURCE_NOTICES.md` и `portfolio_provenance.csv`.

`malyshev_v36_partial.csv.gz` — предыдущий снимок
[публичного ноутбука Neighbour Transfer + Squeeze, версия 36](https://www.kaggle.com/code/malyshevdanil/squares-in-squares-neighbour-transfer-squeeze?scriptVersionId=362914762).
Он получен из опубликованного `submission.csv`: сохранены только 547 раскладок,
у которых сторона меньше стороны обычной решётки. В архиве 290 670 строк;
при отсутствии `n` генератор сам создаёт решётку. Полный CSV содержит
500 500 строк и даёт локальный балл `22.285051125530`; сжатый снимок
воспроизводит тот же балл. Все его раскладки проверены локальным валидатором.
SHA-256 исходного полного CSV:
`6e3497709581354d51d7b7053c49cc261b37d3c57570926b167321c4c8c33233`.
SHA-256 сжатого снимка указан в таблице ниже.

`malyshev_replay_v34.csv` — 60 полных раскладок для отдельных `n`, извлечённых
без оптимизации из `REPLAY_B64` в публичном
[ноутбуке Malyshev Danil, версия 34](https://www.kaggle.com/code/malyshevdanil/squares-in-squares-neighbour-transfer-squeeze).
Числа восстановлены как исходные `float64` из массива, который ноутбук
декодирует через Base64 и LZMA; CSV использует `repr(float)` для обратимого
десятичного представления. В нём 38 215 квадратов. Все 60 раскладок прошли
локальную проверку. Эти координаты являются публичным внешним источником,
отдельным от результатов нашего генератора; в самом ноутбуке указаны его
исходные публичные зависимости и авторы. SHA-256 CSV:
`0dfde02155ad08a0c93c5c743f88233decdd543f3a5599708bc620a499751eec`.

`catalog/parsed-50.txt`, `parsed-54.txt`, `parsed-55.txt` и `parsed-68.txt`
содержат координаты, извлечённые из соответствующих
[`square-N.svg`](https://kingbird.myphotos.cc/packing/squares_in_squares.html)
программой [parse_svg_packing.py](https://github.com/Davidebyzero/packing_squares_in_squares__tools/blob/master/parse_svg_packing.py)
при точности 15 десятичных знаков. В этом прогоне тригонометрические функции
парсера вычислялись стандартным Python `float`; перед добавлением в CSV
координаты масштабируются для малого зазора и проходят строгую проверку
непересечения.

`xu_records.json` — скачиваемый снимок
[`square_packing_records.json`](https://github.com/ry-xu/square_packing/blob/8dc415296f697f5140caea27c7a0193d52deb4e6/square_packing_records.json)
на указанном коммите. Он не входит в Git; команда загрузки есть в корневом
README. Импортёр берёт из него только `n`, сторону, координаты и углы.

Для `n=204`, `227`, `229`, `260` использованы
[архивы координат Evan Daniel](https://github.com/evand/square-packing/tree/0075967bace18a3deabcc0baa64c2fb294fc80a0/search/regularize/lists/certs)
на фиксированном коммите. Углы в исходном JSON указаны в градусах; при импорте
они переведены в радианы. Каждая раскладка проверена на отсутствие пересечений
после добавления численного зазора. Авторство исходных конструкций записано в
`record_seeds.json`; для `n=84` и `n=130` там же сохранены координаты,
полученные локальным линейным поиском.

SHA-256 исходных файлов и сохранённых снимков:

| Файл | SHA-256 |
| --- | --- |
| `ntumlnoob_2268_partial.csv.gz` | `3b35c52c836c18a3be18dc143ee026366b7a293c7091d0c75bc17e19628bad38` |
| `malyshev_v36_partial.csv.gz` | `148105da8592c2f7cb9f63ace6649a87b353840788f8ed3dd6b6796b094f6709` |
| `square-50.svg` | `c044b04db21d01f4a5a6066b23bffe89f9585c4b99c88c4cca6b13af7b692c95` |
| `square-54.svg` | `96afd34f230d10c5dc750b8209fecb90bbebc01f4519cf58193051b9b7ddcaec` |
| `square-55.svg` | `77a0ff9aee57df261c922f45250d7b17b8fdc063c63be3a9a1c0772eb85eb3f7` |
| `square-68.svg` | `8387ef9c17e77c13c00bb3a62b01784a57e567a7a42fc13c1028e2690c0ef6ce` |
| `parsed-50.txt` | `5324de71827cd92fe901fff08f6e4af3a31c581700759807d9d3d26e494a5b1f` |
| `parsed-54.txt` | `ef4c8328436d0a2234e260b990f1ecff9902db529e42cf39d98263fd24d31df4` |
| `parsed-55.txt` | `d07431426aaddf3ea45b4e16a7ac8323e13b4d17c29db25e616ab76c073758e1` |
| `parsed-68.txt` | `a595ac1050eaa8fec0c1074e47a2426b094a0d02330b7e59ddb9f98f1b40afd2` |
| `xu_records.json` | `93130ffb2f9da1b15ba5e8e491e60509192d6fb21108aa2adc86d29c4f071e5d` |
| `n-204.json.gz` | `565dd7e6d226ca5ca74c4d83ad11ead8ac765679c72bd5e8a59290e2f8d74de1` |
| `n-227.json.gz` | `2c4965ed1827d3d5c10f914f00aec1d1ac2e692fd26f2afea8f12bf977e406d4` |
| `n-229.json.gz` | `21fc6d7596be3a02987e57febb92f19310d7c2e6c1e3bda98444e63ad97b99cb` |
| `n-260.json.gz` | `cb3dea1d586244cd1bee36cb0a03c7997e5e169f9edc40b41feea1f59f7dc923` |
