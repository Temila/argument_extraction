import sys, xlrd, glob

class Step_one:

    def __init__(self):
        # reload(sys)
        # sys.setdefaultencoding('utf8') Constitutional_monarchy
        self.files = glob.glob("data/wiki12_articles/*")
        self.wb = xlrd.open_workbook('data/2014_7_18_ibm_CDCdata.xls')
        self.topics, self.CDCs = self._get_topics_for_articles()

    def _get_topics_for_articles(self):
        topics= {}
        CDCs = []
        sheet_names = self.wb.sheet_names()
        sheet = self.wb.sheet_by_name(sheet_names[0])
        n_rows = sheet.nrows
        for i in range(n_rows):
            topic = sheet.cell(i,0).value
            title = sheet.cell(i,1).value.replace(' ','_').replace('&', 'and').replace('?','')
            # print topic + ':' + title
            topics[title] = topic
            CDCs.append({'title':title, 'CDC':sheet.cell(i,2).value})
            i += 1
        return topics, CDCs

    def get_CDCs_by_title(self, title):
        CDCs_title = []
        for x in self.CDCs:
            if x['title'] == title:
                # text = x['CDC'].encode('utf8')
                # CDCs_title.append(unicode(text, errors='replace'))
                CDCs_title.append(x['CDC'])
        return CDCs_title