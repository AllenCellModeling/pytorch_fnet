import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argschema


class RegistrationFiguresParameters(argschema.ArgSchema):
    registration_csv = argschema.fields.InputFile(required=True,
        description='csv file with registration results')
    output_figure = argschema.fields.OutputFile(required=True,
        description='path to where to save figures')

if __name__ == '__main__':
    mod = argschema.ArgSchemaParser(schema_type=RegistrationFiguresParameters)

    df = pd.read_csv(mod.args['registration_csv'])
    f,ax = plt.subplots()
    df.plot(kind='hist',y='pixel_error',ax=ax,bins=np.arange(0,10,.25),legend=False)
    ax.set_xlabel('average pixel error')
    ax.set_ylabel('number of tiles')
    plt.savefig(mod.args['output_figure'])
