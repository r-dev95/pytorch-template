<!--
    Sphinxのドキュメントについて
 -->

# Sphinxのドキュメントについて

Sphinxのドキュメントについて、作成手順を示します。

本手順は、ドキュメント作成にあたって、実行した手順を説明するものであり、
Sphinxの使い方を網羅的に説明するものでないことに注意してください。
また分かりやすさのため、ソースやディレクトリ構造は一部省略または改変して説明しています。

Sphinxのより詳細な使い方は[公式サイト](https://www.sphinx-doc.org/en/master/#)を参照してください。

## 必要なパッケージ

下記パッケージを`pip install`してください。

* sphinx
* sphinx-intl
* sphinx-rtd-theme

## 手順

下記のようなディレクトリ構造でカレントディレクトリが`docs`である前提で進めます。

``` none
pj_root/
├── docs # Sphinxでドキュメントを作成するディレクトリ
└── src
    ├── run.py
    └── lib
        ├── __init__.py
        ├── template.py
        └── common
            ├── __init__.py
            └── decorator.py
```

### 1. Sphinxのディレクトリと必要なファイルを作成する

下記のコマンドを実行し、インタラクティブに設定を行います。

``` bash
sphinx-quickstart
```

``` none
...
> Separate source and build directories (y/n) [n]: <yかn>

The project name will occur in several places in the built documentation.
> Project name: <プロジェクト名>
> Author name(s): <著者名>
> Project release []: <リリースバージョン>
> Project language [en]: <言語>
...
```

`docs`ディレクトリの構造が、下記のようになっていればOKです。

``` none
pj_root/docs/
├── Makefile
├── make.bat
├── build
└── source
    ├── conf.py
    ├── index.rst
    ├── _static
    └── _templates
```

下表を設定した前提で進めます。

|項目                                   |設定値                 |
|---------------------------------------|-----------------------|
|`source`と`build`ディレクトリを分けるか|y                      |
|プロジェクト名                         |pj_name                |
|著者名                                 |author_name            |
|リリースバージョン                     |release_version        |
|言語                                   |en                     |

### 2. `sphinx-apidoc`コマンドで必要なテンプレートファイルを作成する

デフォルトのテンプレートファイルでもいいですが、これをベースに編集して使用します。

(デフォルトのテンプレートファイルのパス：`site-packages/sphinx/templates/apidoc/`)

編集したテンプレートファイルは、`source/_template/apidoc/`に置きます。

#### 2-1. `module.rst.jinja`を編集する

例えば、`lib/template.py`の場合、下記コードの`basename`が`lib.template`となります。

このときドキュメント上のサイドバーやヘッダーの表記は下表のようになります。

``` diff
    {%- if show_headings %}
-   {{- [basename, "module"] | join(' ') | e | heading }}
+   {{- [basename.split(".")[-1], ".py"] | join('') | e | heading }}

    {% endif -%}
    .. automodule:: {{ qualname }}
    {%- for option in automodule_options %}
    :{{ option }}:
    {%- endfor %}
```

|編集前の表記                    |編集後の表記                    |
|:------------------------------:|:------------------------------:|
|`lib.template module`           |`template.py`                   |

#### 2-2. `packages.rst.jinja`を編集する

下記の編集は、`package`と`namespace`のパターンで、
`module.rst.jinja`の例と似ているので編集前後の表記は省略します。

``` diff
    ...
    {%- if is_namespace %}
-   {{- [pkgname, "namespace"] | join(" ") | e | heading }}
+   {{- [pkgname.split(".")[-1], "namespace"] | join(" ") | e | heading }}
    {% else %}
-   {{- [pkgname, "package"] | join(" ") | e | heading }}
+   {{- [pkgname.split(".")[-1], "package"] | join(" ") | e | heading }}
    {% endif %}
    ...
```

次に下記の編集で、`Subpackages`と`Submodules`という表記をサイドバーやヘッダーから削除しています。

``` diff
    {%- if subpackages %}
-   Subpackages
-   -----------
    ...
    {%- if submodules %}
-   Submodules
-   ----------
```

最後の編集は、`module.rst.jinja`の例と同様です。

``` diff
    ...
    {% if separatemodules %}
    {{ toctree(submodules) }}
    {% else %}
    {%- for submodule in submodules %}
    {% if show_headings %}
-   {{- [submodule, "module"] | join(" ") | e | heading(2) }}
+   {{- [submodule.split(".")[-1], ".py"] | join("") | e | heading(2) }}
    {% endif %}
    {{ automodule(submodule, automodule_options) }}
    {% endfor %}
    {%- endif %}
    ...
```

### 3. `.rst`ファイルを作成する

下記のコマンドを実行します。

このときモジュールのパスから``__init__.py``のあるディレクトリを再帰的に解析し、
`.rst`ファイルが作成されます。

``` bash
sphinx-apidoc -o source/ ../src/ -t source/_templates/apidoc/ -d 2 -T -M -f
```

``` bash
# sphinx-apidoc [OPTIONS] -o <出力ディレクトリ> <モジュールのパス>
# [OPTIONS]:
#   -t <パス>       : テンプレートのパス
#   -d <深さ>       : TOC(Table of contents)に表示するサブモジュールの深さ
#   -T              : TOCファイルを作成しない
#   -M              : サブモジュールの前にモジュールのドキュメントを置く。
#   -f              : 上書き
```

`source`ディレクトリの構造が、下記のようになっていればOKです。

``` none
pj_root/docs/source/
├── index.rst
├── run.rst        # 出力されたファイル
├── lib.rst        # 出力されたファイル
├── lib.common.rst # 出力されたファイル
├── _static
└── _templates
    └── apidoc
        ├── module.rst.jinja
        └── package.rst.jinja
```

> [!TIP]
>
> `sphinx-quickstart`は引数からも設定できます。
>
> ``` bash
> sphinx-quickstart --sep -p <プロジェクト名> -a <著者名> -r <リリースバージョン> -l <言語>
> ```

### 4. `index.rst`を編集する

先ほど出力された`.rst`ファイルのベース名を`toctree`のブロックに追加します。

このとき、`lib.common.rst`は`lib.rst`から呼ばれるため、追加していません。

また必要のない文章は削除しています。

``` diff
    pj_name documentation
    =====================

-   Add your content using ``reStructuredText`` syntax. See the
-   `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
-   documentation for details.


    .. toctree::
       :maxdepth: 2
       :caption: Contents:

+      lib
+      run

+   Index:
+   ------

+   * :ref:`genindex`
+   * :ref:`modindex`
```

### 5. `conf.py`を編集する

詳細な説明は省略しますが、`conf.py`から見たモジュールの相対パスを追加する必要があります。

(`sys.path.append(os.path.abspath('../../src/'))`の箇所)

またドキュメントのテーマは`sphinx_rtd_theme`を使用しています。

(`html_theme = 'sphinx_rtd_theme'`の箇所)

``` diff
    # Configuration file for the Sphinx documentation builder.
    #
    # For the full list of built-in configuration values, see the documentation:
    # https://www.sphinx-doc.org/en/master/usage/configuration.html

+   # import standard python modules
+   import os
+   import sys
+   sys.path.append(os.path.abspath('../../src/'))

    # -- Project information -----------------------------------------------------
    # https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

    project = 'pj_name'
    copyright = '2024, author_name'
    author = 'author_name'
    release = 'release_version'

    # -- General configuration ---------------------------------------------------
    # https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

-   extensions = []
+   extensions = [
+       'sphinx.ext.autodoc',
+       'sphinx.ext.napoleon',
+       'sphinx.ext.viewcode',
+       # 'myst_parser',
+   ]

+   # sphinx.ext.autodoc
+   # autoclass_content = 'both' # [class, both, init]
+   autodoc_member_order = 'bysource' # [alphabetical, groupwise, bysource]
+   autodoc_typehints = 'none' # [signature, description, none, both]
+   # sphinx.ext.napoleon
+   napoleon_use_admonition_for_examples = True
+   napoleon_use_admonition_for_notes = True
+   napoleon_use_admonition_for_references = True
+   napoleon_use_ivar = True
+   napoleon_use_param = True
+   napoleon_use_rtype = False
+   napoleon_custom_sections = [('Returns', 'params_style')]

+   # Options for internationalisation
    language = 'en'
+   # Options for markup
+   keep_warnings = True
+   # Options for object signatures
+   toc_object_entries_show_parents = 'hide' # [domain, hide, all]
+   add_module_names = False
+   # Options for templating
    templates_path = ['_templates']
+   # Options for source files
    exclude_patterns = []
+   # Options for the nitpicky mode
+   nitpicky = True
+   nitpick_ignore_regex = [
+       (r'py:class', r'Logger'),
+       (r'py:class', r'logging.Formatter'),
+       (r'py:class', r'Path'),
+       (r'py:class', r'collections.abc.Callable'),
+   ]

    # -- Options for HTML output -------------------------------------------------
    # https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

-   html_theme = 'alabaster'
+   html_theme = 'sphinx_rtd_theme'
    html_static_path = ['_static']
```

### 6. htmlを作成する

下記のコマンドを実行します。
引数は必要に応じて、指定してください。(`SPHINXOPTS='-a -E'`の部分)

すると、`build`ディレクトリに様々なファイルが作成されます。

`index.html`は、`docs/build/html/`にできます。

``` bash
make html SPHINXOPTS='-a -E'
```

**Sphinxドキュメントの作成完了です。**

## 国際化対応(i18n)

上記、手順で作成したドキュメントをベースに他の言語でドキュメントを作成します。

### 1. `conf.py`を編集する

`conf.py`に下記を追加します。

``` python
gettext_compact = False
locale_dirs = ['locale/']
```

### 2. `.pot`ファイルを作成する

下記コマンドで`.pot`ファイルを作成します。

`.pot`ファイルは、`docs/build/gettext/`に作成されます。

``` bash
cd docs
make gettext
```

### 3. `.po`ファイルを作成する

下記コマンドで`.po`ファイルを作成します。
`-l`の引数には、作成する言語を指定します。

`.pot`ファイルは、`docs/source/locale/ja/LC_MESSAGES/`に作成されます。

``` bash
sphinx-intl update -p build/gettext -l ja
```

### 4. `.po`ファイルを編集する

作成された`.po`ファイルに翻訳を追加します。

`#: ...`が対象で、`msgid "..."`の翻訳を`msgstr "***"`の`***`に記載します。

記載しない場合、`msgid "..."`がそのまま適用されます。

``` diff
    # SOME DESCRIPTIVE TITLE.
    # Copyright (C) 2024, author_name
    # This file is distributed under the same license as the pj_name package.
    # FIRST AUTHOR <EMAIL@ADDRESS>, YEAR.
    #
    #, fuzzy
    msgid ""
    msgstr ""
    "Project-Id-Version: pj_name \n"
    "Report-Msgid-Bugs-To: \n"
    "POT-Creation-Date: 2024-11-12 13:18+0900\n"
    "PO-Revision-Date: YEAR-MO-DA HO:MI+ZONE\n"
    "Last-Translator: FULL NAME <EMAIL@ADDRESS>\n"
    "Language-Team: LANGUAGE <LL@li.org>\n"
    "MIME-Version: 1.0\n"
    "Content-Type: text/plain; charset=UTF-8\n"
    "Content-Transfer-Encoding: 8bit\n"

    #: ../../source/index.rst:9
    msgid "Contents:"
    msgstr ""

    #: ../../source/index.rst:7
    msgid "Welcome pj_name documentation!"
-   msgstr ""
+   msgstr "pj_nameドキュメントへようこそ!"
```

### 5. htmlを作成する

下記のコマンドを実行します。

``` bash
make html -e SPHINXOPTS='-a -E -D language="ja"'
```

> [!WARNING]
>
> デフォルトのまま`Makefile`を使用すると、ソースディレクトリと出力ディレクトリが固定されるため、既にある`build/html/`や`build/doctrees/`を移動させるか、`sphinx-build`コマンドを直接実行するか、もしくは`Makefile`を修正してください。

**他の言語のドキュメントの作成完了です。**
