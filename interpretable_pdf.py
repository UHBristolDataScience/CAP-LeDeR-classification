from fpdf import FPDF
import numpy as np
from nltk.stem import WordNetLemmatizer
import re

from explainability import (get_ti_feature_contributions_for_instance_i,
                            run_tree_interpreter)


class InterpretablePDF:
    """
    Class to produce formatted vignettes that indicate which text elements are contributing to
    any given classification.

    Currently this class only works with the CAP Prostate Cancer data, because the formatting
    corresponds to the unique structure of this text data. However, it could easily be adapted to
    work with other textual data (such as LeDeR).
    """

    def __init__(self,
                 classifier,
                 x_data,
                 y_data,
                 feature_columns,
                 base_font_size=12, line_height=8,
                 header_col_width=100, legend_offset=47.5,
                 legend_offset_2=63,
                 top_n_features=None,
                 contributions=None):

        self.font_size = base_font_size
        self.line_height = line_height
        self.header_col_width = header_col_width
        self.legend_offset = legend_offset
        self.legend_offset_2 = legend_offset_2

        self.top_n = top_n_features

        self.pdf = None
        self.original_data = None

        self.feature_columns = feature_columns
        self.X = x_data
        self.y = y_data
        self.clf = classifier
        _, _, self.contributions = (run_tree_interpreter(self.clf,
                                                         self.X)
                                    if contributions is None else (
                                            None, None, contributions))

        self.stemmer = WordNetLemmatizer()

        self.section_headers = ['Clinical features at diagnosis',
                                'Treatments received',
                                'Prostate cancer progression',
                                'Progression of co-morbidities',
                                'End of life']

        self.vignette_column_names = [
            'Gleason Score at diagnosis (with dates)',
            'Clinical stage (TNM)',
            'Pathological stage (TNM)',
            'Co-morbidities with dates of diagnosis',
            'Other primary cancers with dates of diagnosis',
            'PSA level at diagnosis with dates',
            'Radiological evidence of local spread at diagnosis',
            'Radiological evidence of metastases at diagnosis',
            'Initial treatments (dates)',
            'Hormone therapy (start date)',
            'Maximum androgen blockade (start date)',
            'Orchidectomy (date)',
            'Chemotherapy (start date)',
            'Treatment for complications of treating prostate cancer with dates (if available)',
            'Serial PSA levels (dates)',
            'Serum testosterone',
            'Radiological evidence of metastases',
            'Other indications or complications of disease progression',
            'Date of recurrence following radical surgery or radiotherapy',
            'Palliative care referrals and treatments',
            'Treatment/ admission for co-morbidity with dates (if available)',
            'Symptoms in last 3-6 months (i.e. bone pain, weight loss, cachexia,\
               loss of appetite, obstructive uraemia)',
            'Last consultation: speciality & date',
            'Was a DS1500 report issued?',
            'Post mortem findings']

    def create_pdf(self, case_id, original_data, filename):

        self.original_data = original_data

        self.pdf = FPDF()
        self.pdf.add_page()
        self.pdf.set_font('Arial', 'B', self.font_size)

        self.pdf.cell(w=0,
                      h=self.line_height,
                      txt='Interpretable Vignette Classification for Cause of Death Review',
                      border=0,
                      ln=0,
                      align='C', fill=False, link='')

        self.pdf.set_font('')
        self.pdf.ln()
        self.pdf.ln()

        y = self.pdf.get_y()
        self.pdf.multi_cell(w=self.header_col_width,
                            h=self.line_height,
                            txt='Study ID number : %s\nDate of death : %s\nDate of diagnosis : %s' % (
                                original_data['cp1random_id_5_char'],
                                original_data['cnr19datedeath'].date(),
                                original_data['cnr_date_pca_diag'].date()),
                            border=0,
                            align='L',
                            fill=False)
        self.pdf.y = y
        self.pdf.x = self.header_col_width
        self.pdf.multi_cell(w=self.header_col_width,
                            h=self.line_height,
                            txt='Predicted death code: %d (%.2f)\nActual death code: %d\nCOD route: %d' % (
                                self.clf.best_estimator_.predict(self.X)[case_id],
                                self.clf.best_estimator_.predict_proba(self.X)[case_id][1],
                                original_data.pca_death_code,
                                original_data.cp1do_cod_route),
                            border=0,
                            align='R', fill=False)

        self.pdf.set_line_width(0.5)
        self.pdf.line(10, self.pdf.get_y(), 210 - 10, self.pdf.get_y())
        self.pdf.set_line_width(0.2)

        self.write_legend(case_id)

        for ci, col in enumerate(self.feature_columns):

            self.pdf.set_text_color(0, 0, 0)
            self.pdf.set_font_size(self.font_size)

            if ci in [0, 8, 14, 20, 21]:
                self.print_section_header(0)

            self.pdf.line(10, self.pdf.get_y(), 210 - 10, self.pdf.get_y())
            self.pdf.write(self.line_height, self.vignette_column_names[ci] + ': ')

            if 'palliative' not in col:
                text = str(original_data[col])
                self.print_paragraph(text, case_id)

        self.pdf.output(filename)

    def get_font_size_and_color(self,
                                contribution,
                                min_contribution,
                                max_contribution,
                                shrink=0.5):
        c = (255, 160, 0) if contribution < 0 else (0, 0, 255)
        s = (
                (self.font_size * shrink) + 1.5 * self.font_size
                * (np.absolute(contribution) - min_contribution)
                / float(max_contribution - min_contribution)
        )

        return s, c

    def legend_entry(self, fimps, fimp, align):

        size, color = self.get_font_size_and_color(fimp.contribution,
                                                   fimps.magnitude.min(),
                                                   fimps.magnitude.max())
        self.pdf.set_text_color(*color)
        self.pdf.set_font_size(size)
        self.pdf.cell(w=self.legend_offset,
                      h=self.line_height,
                      txt=fimp.feature,
                      border=0, ln=0,
                      align=align, fill=False, link='')

    def legend_label(self, text, align):

        self.pdf.cell(w=self.legend_offset_2,
                      h=self.line_height,
                      txt=text, border=0, ln=0,
                      align=align, fill=False, link='')

    def write_legend(self, case_id):

        self.pdf.cell(w=0,
                      h=self.line_height,
                      txt='Feature contribution legend',
                      border=0, ln=0,
                      align='C', fill=False, link='')
        self.pdf.ln()

        fimps = get_ti_feature_contributions_for_instance_i(case_id,
                                                            self.contributions,
                                                            self.clf).sort_values(by='magnitude',
                                                                                  ascending=False)
        fimps = fimps.head(self.top_n) if self.top_n is not None else fimps

        fimp = fimps.loc[fimps.contribution.idxmin()]
        self.legend_entry(fimps, fimp, align='L')

        fimp = fimps.loc[fimps.contribution < 0]
        fimp = fimp.loc[fimp.contribution.idxmax()]
        self.legend_entry(fimps, fimp, align='R')

        fimp = fimps.loc[fimps.contribution > 0]
        fimp = fimp.loc[fimp.contribution.idxmin()]
        self.legend_entry(fimps, fimp, align='L')

        fimp = fimps.loc[fimps.contribution.idxmax()]
        self.legend_entry(fimps, fimp, align='R')

        self.pdf.ln()
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.set_font('Arial', '', self.font_size * .6)

        self.legend_label('Largest negative contribution', 'L')
        self.legend_label('Smallest contributions', 'C')
        self.legend_label('Largest positive contribution', 'R')

        self.pdf.set_font('Arial', '', self.font_size)

    def print_section_header(self, section):

        self.pdf.ln()
        self.pdf.set_line_width(0.5)
        self.pdf.line(10, self.pdf.get_y(), 210 - 10, self.pdf.get_y())
        self.pdf.set_line_width(0.2)
        self.pdf.set_font('Arial', 'B', self.font_size)
        self.pdf.write(self.line_height, self.section_headers[section] + '\n')
        self.pdf.set_font('')

    # REFACTOR!
    def print_paragraph(self, text, i,
                        base_color=(128, 128, 128)):

        fimps = get_ti_feature_contributions_for_instance_i(i,
                                                            self.contributions,
                                                            self.clf)
        fimps.sort_values(by='magnitude', inplace=True, ascending=False)
        fimps = fimps.head(self.top_n)

        old_word = ''
        old_tr_word = ''
        old_bigram = ''
        old_bigram_contribution = None

        old_color = base_color
        old_size = self.font_size

        words = text.split(' ')
        words.append('  .')

        for word in words:
            tr_word = self.transform_text(word)

            feat_tr = old_tr_word + ' ' + tr_word
            contribution_bi = (fimps.loc[fimps.feature == feat_tr]
                               .iloc[0].contribution
                               if feat_tr in list(fimps.feature)
                               else None)
            magnitude_bi = np.absolute(contribution_bi) if contribution_bi is not None else 0

            feat_tr = old_tr_word
            contribution_uni = (fimps.loc[fimps.feature == feat_tr]
                                .iloc[0].contribution
                                if feat_tr in list(fimps.feature)
                                else None)
            magnitude_uni = np.absolute(contribution_uni) if contribution_uni is not None else 0

            if contribution_bi and magnitude_bi > magnitude_uni:
                # print('bigram: ', old_tr_word)
                feat_tr = old_tr_word + ' ' + tr_word
                feat = old_word + ' ' + word

                contribution = fimps.loc[fimps.feature == feat_tr].iloc[0].contribution
                size, color = self.get_font_size_and_color(contribution,
                                                           fimps.magnitude.min(),
                                                           fimps.magnitude.max())
                self.pdf.set_text_color(*color)
                self.pdf.set_font_size(size)
                feat = feat.encode('latin-1', 'replace').decode('latin-1')
                self.pdf.write(self.line_height, feat + ' ')

                old_word = ''
                old_tr_word = ''
                old_color = base_color
                old_size = self.font_size

            elif contribution_uni and magnitude_uni > magnitude_bi:
                # print('unigram: ', old_tr_word)
                feat_tr = old_tr_word
                feat = old_word

                contribution = fimps.loc[fimps.feature == feat_tr].iloc[0].contribution
                size, color = self.get_font_size_and_color(contribution,
                                                           fimps.magnitude.min(),
                                                           fimps.magnitude.max())
                self.pdf.set_text_color(*color)
                self.pdf.set_font_size(size)
                feat = feat.encode('latin-1', 'replace').decode('latin-1')
                self.pdf.write(self.line_height, feat + ' ')

                old_word = word
                old_tr_word = tr_word
                old_color = base_color
                old_size = self.font_size

            else:
                self.pdf.set_text_color(*old_color)
                self.pdf.set_font_size(old_size)
                w = old_word.encode('latin-1', 'replace').decode('latin-1')
                self.pdf.write(self.line_height, w + ' ')

                old_word = word
                old_tr_word = tr_word
                old_color = base_color
                old_size = self.font_size

        self.pdf.ln()

    def transform_text(self, text):

        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(text))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        # document = re.sub(r'\s+[a-zA-Z]\s+', ' ', str(X[sen]))

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()

        # Lemmatization
        document = document.split()

        document = [self.stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)

        return document
