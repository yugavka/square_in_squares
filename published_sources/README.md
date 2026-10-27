# Источники импортированных раскладок

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

SHA-256 исходных файлов и сохранённых снимков:

| Файл | SHA-256 |
| --- | --- |
| `square-50.svg` | `c044b04db21d01f4a5a6066b23bffe89f9585c4b99c88c4cca6b13af7b692c95` |
| `square-54.svg` | `96afd34f230d10c5dc750b8209fecb90bbebc01f4519cf58193051b9b7ddcaec` |
| `square-55.svg` | `77a0ff9aee57df261c922f45250d7b17b8fdc063c63be3a9a1c0772eb85eb3f7` |
| `square-68.svg` | `8387ef9c17e77c13c00bb3a62b01784a57e567a7a42fc13c1028e2690c0ef6ce` |
| `parsed-50.txt` | `5324de71827cd92fe901fff08f6e4af3a31c581700759807d9d3d26e494a5b1f` |
| `parsed-54.txt` | `ef4c8328436d0a2234e260b990f1ecff9902db529e42cf39d98263fd24d31df4` |
| `parsed-55.txt` | `d07431426aaddf3ea45b4e16a7ac8323e13b4d17c29db25e616ab76c073758e1` |
| `parsed-68.txt` | `a595ac1050eaa8fec0c1074e47a2426b094a0d02330b7e59ddb9f98f1b40afd2` |
| `xu_records.json` | `93130ffb2f9da1b15ba5e8e491e60509192d6fb21108aa2adc86d29c4f071e5d` |
