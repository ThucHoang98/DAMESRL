from time import gmtime, strftime

from  DAMESRL.liir.dame.core.io.CoNLL2005Reader import CoNLL2005Reader
from  DAMESRL.liir.dame.core.representation.Predicate import Predicate
from  DAMESRL.liir.dame.core.representation.Text import Text


class HTMLWriter:
    def write_props(self, txt, output):
        if isinstance(txt, Text) or isinstance(txt, list):
            f = open(output, "w")
            html_preds, num_sentences, num_preds, tags = self.get_html_preds(txt)
            tag_colors = {t: list(Colors.cnames.values())[i] for i, t in enumerate(tags)}
            tag_colors['V'] = Colors.cnames['red']
            tag_colors['O'] = 'white'

            tag_colors['A0'] = Colors.cnames['blue']
            tag_colors['A1'] = Colors.cnames['orange']
            tag_colors['A5'] = Colors.cnames['olive']
            tag_colors['C-A1'] = Colors.cnames['gold']
            tag_colors['A2'] = Colors.cnames['purple']
            tag_colors['AM-TMP'] = Colors.cnames['green']
            tag_colors['AM-LOC'] = Colors.cnames['brown']
            tag_colors['AM-ADV'] = Colors.cnames['coral']
            tag_colors['AM-MOD'] = Colors.cnames['cyan']
            tag_colors['AM-CAU'] = Colors.cnames['steelblue']
            tag_colors['AM-DIS'] = Colors.cnames['darkmagenta']

            self.write_header(f, num_sentences, num_preds, tag_colors)
            f.write(html_preds)
            self.write_footer(f)

    def simplify_tag(self, tag):
        return tag.replace('B-', '').replace('I-', '')

    def get_html_preds(self, txt):
        num_preds, tags, html = 0, set(), ""
        for s_i, s in enumerate(txt):
            preds = [w for w in s if isinstance(w, Predicate)]
            html += "<div class=sentence><h2>S" + str(s_i) + "</h2>\n<table>\n"
            for pred in preds:
                num_preds += 1
                tags = tags.union(pred.arguments)
                html += "<tr>"
                html += self.pred_to_html_row(pred)
                html += "</tr\n>"

            html += self.sentence_to_html_row(s)

            html += "</table>\n</div>"
        return html, len(txt), num_preds, [self.simplify_tag(t) for t in tags]

    def pred_to_html_row(self, p):
        html = "<tr>"
        prev = None
        for tag in p.arguments:
            tag = self.simplify_tag(tag)
            viz_tag = '' if tag == 'O' else ' ' if tag == prev else tag
            html += '<td class="' + tag + '">' + viz_tag + "</td>"
            prev = tag
        html += "<td class=lemma>&larr; " + p.lemma + "</td>"
        html += "</tr>"
        return html

    def sentence_to_html_row(self, s):
        html = "<tr>"
        for word in s:
            html += "<td class=word>" + word.form + "</td>"
        html += "</tr>"
        return html

    def write_header(self, f, num_sentences, num_preds, tag_colors):  # Write HTML Header
        header = """
        <!DOCTYPE html>
<html lang="en"><head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<style>
body { max-width: 95%; margin: 0 auto;  font-family: Georgia, serif; line-height: 1.5; }
h1 { margin: 2rem 0; font-size: 1.6rem; line-height: 1.2; }
h2 { margin: 2rem 0 1.25rem; font-size: 1.4rem; font-family: Avenir Next, Avenir, sans-serif; font-weight: 400; line-height: 1.3; }
h3 { margin: 1.65rem 0 1rem; font-size: 1.1rem; line-height: 1.4; }
table { margin: 1.5rem 0; border-collapse: collapse; border-bottom: solid 1px rgba(0,0,0,.1); white-space: nowrap;}
.lemma {border-color:white;}
.word {font-style: italic;}
.tag {color:white;}
td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-top-width:1px;border-bottom-width:1px;text-align:center;}
""" + "\n".join(
            ["." + str(t) + "{border-color:" + str(tag_colors[t]) + ";background:" + str(tag_colors[t]) + ";}" for t in
             tag_colors]) + """
</style>
<title>SRL</title>
</head>
<body>
<h1>
Dame SRL Output
</h1>
<p><time>""" + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + """</time> • """ + str(
            num_sentences) + """ sentences containing """ + str(num_preds) + """ predicates</p>
<br>
"""
        f.write(header)

    def write_footer(self, f):
        f.write("</body>\n</html>")


class Colors:
    cnames = {
        'aliceblue': '#F0F8FF',
        'antiquewhite': '#FAEBD7',
        'aqua': '#00FFFF',
        'aquamarine': '#7FFFD4',
        'azure': '#F0FFFF',
        'bisque': '#FFE4C4',
        'blanchedalmond': '#FFEBCD',
        'blue': '#0000FF',
        'blueviolet': '#8A2BE2',
        'brown': '#A52A2A',
        'burlywood': '#DEB887',
        'cadetblue': '#5F9EA0',
        'chartreuse': '#7FFF00',
        'chocolate': '#D2691E',
        'coral': '#FF7F50',
        'cornflowerblue': '#6495ED',
        'cornsilk': '#FFF8DC',
        'crimson': '#DC143C',
        'cyan': '#00FFFF',
        'darkblue': '#00008B',
        'darkcyan': '#008B8B',
        'darkgoldenrod': '#B8860B',
        'darkgray': '#A9A9A9',
        'darkgreen': '#006400',
        'darkkhaki': '#BDB76B',
        'darkmagenta': '#8B008B',
        'darkolivegreen': '#556B2F',
        'darkorange': '#FF8C00',
        'darkorchid': '#9932CC',
        'darkred': '#8B0000',
        'darksalmon': '#E9967A',
        'darkseagreen': '#8FBC8F',
        'darkslateblue': '#483D8B',
        'darkslategray': '#2F4F4F',
        'darkturquoise': '#00CED1',
        'darkviolet': '#9400D3',
        'deeppink': '#FF1493',
        'deepskyblue': '#00BFFF',
        'dimgray': '#696969',
        'dodgerblue': '#1E90FF',
        'firebrick': '#B22222',
        'forestgreen': '#228B22',
        'fuchsia': '#FF00FF',
        'gainsboro': '#DCDCDC',
        'gold': '#FFD700',
        'goldenrod': '#DAA520',
        'gray': '#808080',
        'green': '#008000',
        'greenyellow': '#ADFF2F',
        'hotpink': '#FF69B4',
        'indianred': '#CD5C5C',
        'indigo': '#4B0082',
        'khaki': '#F0E68C',
        'lavender': '#E6E6FA',
        'lavenderblush': '#FFF0F5',
        'lawngreen': '#7CFC00',
        'lemonchiffon': '#FFFACD',
        'lightblue': '#ADD8E6',
        'lightcoral': '#F08080',
        'lightgreen': '#90EE90',
        'lightgray': '#D3D3D3',
        'lightpink': '#FFB6C1',
        'lightsalmon': '#FFA07A',
        'lightseagreen': '#20B2AA',
        'lightskyblue': '#87CEFA',
        'lightslategray': '#778899',
        'lightsteelblue': '#B0C4DE',
        'lime': '#00FF00',
        'limegreen': '#32CD32',
        'linen': '#FAF0E6',
        'magenta': '#FF00FF',
        'maroon': '#800000',
        'mediumaquamarine': '#66CDAA',
        'mediumblue': '#0000CD',
        'mediumorchid': '#BA55D3',
        'mediumpurple': '#9370DB',
        'mediumseagreen': '#3CB371',
        'mediumslateblue': '#7B68EE',
        'mediumspringgreen': '#00FA9A',
        'mediumturquoise': '#48D1CC',
        'mediumvioletred': '#C71585',
        'midnightblue': '#191970',
        'mistyrose': '#FFE4E1',
        'moccasin': '#FFE4B5',
        'navy': '#000080',
        'oldlace': '#FDF5E6',
        'olive': '#808000',
        'olivedrab': '#6B8E23',
        'orange': '#FFA500',
        'orangered': '#FF4500',
        'orchid': '#DA70D6',
        'palegoldenrod': '#EEE8AA',
        'palegreen': '#98FB98',
        'paleturquoise': '#AFEEEE',
        'palevioletred': '#DB7093',
        'papayawhip': '#FFEFD5',
        'peachpuff': '#FFDAB9',
        'peru': '#CD853F',
        'pink': '#FFC0CB',
        'plum': '#DDA0DD',
        'powderblue': '#B0E0E6',
        'purple': '#800080',
        'red': '#FF0000',
        'rosybrown': '#BC8F8F',
        'royalblue': '#4169E1',
        'saddlebrown': '#8B4513',
        'salmon': '#FA8072',
        'sandybrown': '#FAA460',
        'seagreen': '#2E8B57',
        'sienna': '#A0522D',
        'silver': '#C0C0C0',
        'skyblue': '#87CEEB',
        'slateblue': '#6A5ACD',
        'slategray': '#708090',
        'springgreen': '#00FF7F',
        'steelblue': '#4682B4',
        'tan': '#D2B48C',
        'teal': '#008080',
        'thistle': '#D8BFD8',
        'tomato': '#FF6347',
        'turquoise': '#40E0D0',
        'violet': '#EE82EE',
        'wheat': '#F5DEB3',
        'yellowgreen': '#9ACD32'}


if __name__ == "__main__":
    import sys

    writer = HTMLWriter()
    reader = CoNLL2005Reader(sys.argv[1])

    writer.write_props(reader.read_all(), sys.argv[2])
