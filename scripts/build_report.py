"""Genera el informe PDF desde Markdown y CSV; requiere requirements-report.txt."""
import csv
import json
import re
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'
ASSETS = DOCS / 'assets'


def figures():
    import os
    (ROOT / 'build').mkdir(exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "build/matplotlib-cache"))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    green, ink = '#008A36', '#21352C'
    for final in [False, True]:
        fig, ax = plt.subplots(figsize=(9, 3.15))
        ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
        def box(x, y, w, h, label, accent=False):
            ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.015',
                         edgecolor=green if accent else '#A4B9AD', facecolor='#EDF7F0' if accent else '#F5F7F6', linewidth=1.2))
            ax.text(x+w/2,y+h/2,label,ha='center',va='center',color=ink,fontsize=9.3,linespacing=1.5)
        def arrow(a,b):
            ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,color=green,linewidth=1.4))
        box(.025,.65,.245,.24,'CPU: imagen PGM\n800 × 600')
        box(.025,.15,.245,.24,'CPU: precálculo\nsenos y cosenos')
        box(.365,.65,.245,.24,'GPU: imagen\nmemoria global')
        box(.365,.15,.245,.24,'GPU: tablas constantes\n90 + 90 floats · 720 B',True)
        if final:
            box(.70,.46,.275,.43,'BLOQUE DE 256 HILOS\nlocalAcc compartido · 36 000 B\ninicializar → barrera\nvotar → barrera',True)
        else:
            box(.70,.52,.275,.35,'KERNEL\nun hilo por píxel\nlectura de cos[t] y sin[t]',True)
        box(.70,.045,.275,.235,'Acumulador global\n9 000 bins → validación CPU')
        arrow((.28,.77),(.35,.77)); arrow((.28,.27),(.35,.27))
        arrow((.62,.77),(.685,.72)); arrow((.62,.27),(.685,.56))
        arrow((.84,.44 if final else .50),(.84,.30))
        ax.text(.31,.84,'cudaMemcpy',ha='center',fontsize=7.5,color=ink)
        ax.text(.315,.08,'cudaMemcpyToSymbol',ha='center',fontsize=7.5,color=ink)
        name='memoria-final-informe.png' if final else 'memoria-constante-informe.png'
        fig.savefig(ASSETS/name,dpi=220,bbox_inches='tight',facecolor='white');plt.close(fig)
    root=ROOT/'results/fase6-tres-memorias/final'
    filenames=['global-times.csv','constante-times.csv','tres-memorias-times.csv']
    values=[[float(r['kernel_ms']) for i in range(1,7) for r in csv.DictReader((root/f'ronda{i}'/name).open())] for name in filenames]
    fig,ax=plt.subplots(figsize=(8,3.35))
    box=ax.boxplot(values,patch_artist=True,showmeans=True,
        meanprops=dict(marker='D',markerfacecolor=ink,markeredgecolor=ink,markersize=4),
        medianprops=dict(color=ink,linewidth=1.4),widths=.42)
    for patch,color in zip(box['boxes'],['#D9E4DD','#A7D9B7','#3C9D62']):patch.set_facecolor(color);patch.set_edgecolor(green)
    for i,rows in enumerate(values,1):
        ax.text(i,max(rows)+.008,f'{sum(rows)/len(rows):.6f} ms',ha='center',fontsize=9,color=ink)
    ax.set_xticks([1,2,3],['Global','Global + constante','Global + constante\n+ compartida'])
    ax.set_ylabel('Tiempo del kernel (ms)');ax.set_ylim(.70,.955)
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18)
    fig.tight_layout();fig.savefig(ASSETS/'tiempos-informe.png',dpi=220,facecolor='white');plt.close(fig)


def build():
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer,
          PageBreak, Table, TableStyle, Image, Flowable, KeepTogether, Preformatted)
    from reportlab.platypus.tableofcontents import TableOfContents
    from PIL import Image as PILImage, ImageChops
    fontdir=Path('/System/Library/Fonts/Supplemental')
    if (fontdir/'Arial.ttf').exists():
        pdfmetrics.registerFont(TTFont('Body',str(fontdir/'Arial.ttf')))
        pdfmetrics.registerFont(TTFont('BodyBold',str(fontdir/'Arial Bold.ttf')))
        pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='Body',boldItalic='BodyBold')
        normal,bold='Body','BodyBold'
    else:
        from matplotlib.font_manager import findfont, FontProperties
        pdfmetrics.registerFont(TTFont('Body', findfont(FontProperties(family='DejaVu Sans'))))
        pdfmetrics.registerFont(TTFont('BodyBold', findfont(FontProperties(family='DejaVu Sans', weight='bold'))))
        pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='Body',boldItalic='BodyBold')
        normal,bold='Body','BodyBold'
    config=json.loads((DOCS/'informe-config.json').read_text())
    green=colors.HexColor('#008A36'); ink=colors.HexColor('#21352C'); muted=colors.HexColor('#58675F')
    W,H=letter; margin=54; width=W-2*margin
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle('Text',fontName=normal,fontSize=10.0,leading=14.0,textColor=ink,alignment=TA_JUSTIFY,spaceAfter=9))
    styles.add(ParagraphStyle('TitleMain',fontName=bold,fontSize=19,leading=23,textColor=green,spaceAfter=15,keepWithNext=True))
    styles.add(ParagraphStyle('Sub',fontName=bold,fontSize=11.3,leading=15,textColor=ink,spaceBefore=8,spaceAfter=7,keepWithNext=True))
    styles.add(ParagraphStyle('ReferenceText',parent=styles['Text'],alignment=TA_LEFT))
    styles.add(ParagraphStyle('SmallText',fontName=normal,fontSize=8.2,leading=11,textColor=muted,spaceAfter=8))
    styles.add(ParagraphStyle('Cell',fontName=normal,fontSize=8.4,leading=11,textColor=ink,splitLongWords=True))
    styles.add(ParagraphStyle('CellHead',parent=styles['Cell'],fontName=bold,textColor=colors.white))
    styles.add(ParagraphStyle('MonoCode',fontName='Courier',fontSize=7.1,leading=10,backColor=colors.HexColor('#F3F6F4'),borderPadding=10,spaceBefore=5,spaceAfter=12))
    styles.add(ParagraphStyle('AppendixCell',parent=styles['Cell'],fontSize=8,leading=10))
    styles.add(ParagraphStyle('AppendixHead',parent=styles['CellHead'],fontSize=8,leading=10))
    styles.add(ParagraphStyle('AppendixSub',parent=styles['Sub'],fontSize=10,leading=12,spaceBefore=5,spaceAfter=4))
    class Document(BaseDocTemplate):
        def afterFlowable(self,flowable):
            if isinstance(flowable,Paragraph) and flowable.style.name=='TitleMain':
                title=flowable.getPlainText()
                if title=='Índice':return
                key='section-'+str(self.seq.nextf('section'))
                self.canv.bookmarkPage(key);self.canv.addOutlineEntry(title,key,0,False)
                self.notify('TOCEntry',(0,title,self.page,key))
    def footer(c,doc):
        if doc.page==1:return
        c.saveState();c.setStrokeColor(colors.HexColor('#DCE6DF'));c.line(margin,H-40,W-margin,H-40)
        c.setFont(normal,8);c.setFillColor(muted)
        c.drawString(margin,H-30,'UVG · Computación Paralela y Distribuida')
        c.drawRightString(W-margin,H-30,'Proyecto 3 · CUDA')
        c.drawString(margin,29,'Transformada de Hough · Informe final')
        c.drawRightString(W-margin,29,str(doc.page));c.restoreState()
    class Cover(Flowable):
        def __init__(self):super().__init__();self.width=width;self.height=H-112
        def draw(self):
            c=self.canv
            # Se coloca el archivo original omitiendo solo el margen blanco.
            logo=DOCS/'assets/uvg-logo-oficial.png'
            with PILImage.open(logo) as im:
                rgb=im.convert('RGB');bbox=ImageChops.difference(rgb,PILImage.new('RGB',im.size,'white')).getbbox();iw,ih=im.size
            x0,y0,x1,y1=bbox;lw=360;scale=lw/(x1-x0);lh=(y1-y0)*scale;x=(width-lw)/2;y=565
            c.saveState();p=c.beginPath();p.rect(x,y,lw,lh);c.clipPath(p,stroke=0,fill=0)
            c.drawImage(str(logo),x-x0*scale,y-(ih-y1)*scale,width=iw*scale,height=ih*scale,mask='auto');c.restoreState()
            c.setFillColor(muted);c.setFont(normal,11);c.drawCentredString(width/2,515,config['course'])
            c.setStrokeColor(green);c.setLineWidth(1.8);c.line(0,479,width,479)
            c.setFillColor(green);c.setFont(bold,11);c.drawString(0,448,'PROYECTO 3  /  INFORME FINAL')
            c.setFillColor(ink);c.setFont(bold,29)
            c.drawString(0,398,'Transformada de Hough');c.drawString(0,360,'con CUDA')
            c.setFont(normal,12);c.setFillColor(muted)
            c.drawString(0,322,'Evaluación de memoria global, constante y compartida')
            c.setFillColor(ink);c.setFont(bold,10);c.drawString(0,255,'INTEGRANTES');c.drawRightString(width,255,'CARNÉ')
            for i,author in enumerate(config['authors']):
                yy=219-i*35;c.setFont(normal,12);c.drawString(0,yy,author['name']);c.drawRightString(width,yy,author['id'])
                c.setStrokeColor(colors.HexColor('#E6ECE8'));c.setLineWidth(.5);c.line(0,yy-13,width,yy-13)
            c.setFillColor(muted);c.setFont(normal,11);c.drawString(0,55,'Guatemala, '+config['date'])
            c.setFont(normal,8.5);c.drawString(0,31,'Universidad del Valle de Guatemala')
    doc=Document(str(DOCS/'INFORME_FINAL.pdf'),pagesize=letter,leftMargin=margin,rightMargin=margin,
                 topMargin=61,bottomMargin=49,title=config['title'],author=', '.join(a['name'] for a in config['authors']))
    doc.addPageTemplates([PageTemplate(id='body',frames=[Frame(margin,49,width,H-110,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=footer)])
    story=[Cover(),PageBreak(),Paragraph('Índice',styles['TitleMain'])]
    toc=TableOfContents();toc.levelStyles=[ParagraphStyle('TOC',fontName=normal,fontSize=8.8,leading=12.2,textColor=ink,leftIndent=0,firstLineIndent=0,spaceBefore=1)]
    story.extend([toc,Spacer(1,18)])
    def rich(text):
        text=escape(text)
        text=re.sub(r'`([^`]+)`',r'<font name="Courier">\1</font>',text)
        text=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',text)
        # Rutas y URL largas pueden partirse en líneas en las referencias.
        text=re.sub(r'https://[^ ]+',lambda m:'<link href="'+m.group(0)+'" color="#008A36">'+m.group(0)+'</link>',text)
        return text
    lines=(DOCS/'INFORME_FINAL.md').read_text().splitlines();i=0;appendix=False;references=False
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line=='<!-- pagebreak -->':story.append(PageBreak());i+=1;continue
        if line.startswith('# '):
            appendix=line.startswith('# Apéndice')
            references=line.startswith('# 11. Referencias')
            story.append(Paragraph(rich(line[2:]),styles['TitleMain']));i+=1;continue
        if line.startswith('## '):story.append(Paragraph(rich(line[3:]),styles['AppendixSub'] if appendix else styles['Sub']));i+=1;continue
        if line.startswith('```'):
            buf=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'):buf.append(lines[i]);i+=1
            story.append(Preformatted('\n'.join(buf),styles['MonoCode']));i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch('[-: ]+',x) for x in cells):rows.append(cells)
                i+=1
            n=len(rows[0]);widths=None
            if n==6:widths=[width*.32,width*.055,width*.15,width*.175,width*.15,width*.15]
            if n==3:
                widths=[width*.28,width*.42,width*.30] if rows[0][0]=='Caso' else [width*.29,width*.14,width*.57]
            if n==2:widths=[width*.27,width*.73]
            if n==4:widths=[width*.26,width*.2467,width*.2467,width*.2466]
            table=Table([[Paragraph(rich(c),styles['AppendixHead'] if appendix and r==0 else styles['AppendixCell'] if appendix else styles['CellHead'] if r==0 else styles['Cell']) for c in row] for r,row in enumerate(rows)],colWidths=widths,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),green),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#F0F5F2'),colors.white]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),2 if appendix else 4),('BOTTOMPADDING',(0,0),(-1,-1),2 if appendix else 4),('LINEBELOW',(0,-1),(-1,-1),.4,colors.HexColor('#D5E1D9'))]))
            story.extend([table,Spacer(1,8 if appendix else 12)]);continue
        match=re.match(r'!\[(.*?)\]\((.*?)\)',line)
        if match:
            caption,path=match.groups();imagepath=(DOCS/path).resolve()
            if imagepath.suffix=='.pgm':
                converted=ROOT/'build/report-runway.png'
                with PILImage.open(imagepath) as im:im.save(converted)
                imagepath=converted
            with PILImage.open(imagepath) as im:iw,ih=im.size
            fw=width;fh=width*ih/iw
            if 'Figura 4.' in caption:fw=width*.40;fh=fw*ih/iw
            if 'Figura 5.' in caption:fw=width*.68;fh=fw*ih/iw
            story.append(KeepTogether([Image(str(imagepath),width=fw,height=fh),Spacer(1,5),Paragraph(rich(caption),styles['SmallText'])]));i+=1;continue
        buf=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','```','![','<!--')):
            if lines[i].startswith('- ') or re.match(r'\d+\. ',lines[i]):break
            buf.append(lines[i].strip());i+=1
        text=' '.join(buf)
        if text.startswith('- '):text='• '+text[2:]
        story.append(Paragraph(rich(text),styles['ReferenceText'] if references else styles['Text']))
    doc.multiBuild(story)
    print(DOCS/'INFORME_FINAL.pdf')


if __name__=='__main__':
    figures()
    build()
