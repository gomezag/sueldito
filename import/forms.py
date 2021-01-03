from django import forms


class ImportFileForm(forms.Form):
    type = forms.ChoiceField(choices=[('bbvaes', 'BBVA Es'),
                                      ('bbvapy', 'BBVA Py'),
                                      ('bbvapycred', 'BBVA Py Tarj. Credito')])
    file = forms.FileField()