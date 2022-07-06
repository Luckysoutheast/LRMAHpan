from flask import render_template,flash,redirect,url_for,request,send_from_directory,send_file,session
from app import app
from app.program import *
from app.form import QueryForm
import sys
import os
import datetime

@app.route('/result')
def result():
    peptide = request.args.get('peptide')
    mhc = request.args.get('mhc')
    is_neoantigen = request.args.get('neoantigen')  # str
    is_save = request.args.get('is_save')
    is_ma = request.args.get('is_ma')
    print('peptide:\n',peptide)
    print('mhc:\n',mhc,len(mhc))
    if is_ma == 'True':
        print('ResMAHPan predicting......')
        df_ret = ResMHApan_batch(peptide, mhc,is_save)
        return render_template('result2.html',mhc=mhc,p=df_ret['peptide'].to_list(),ba=df_ret['BA'].to_list(),ap=df_ret['AP'].to_list(),ps=df_ret['PS'].to_list(),is_save=is_save, is_neoantigen=is_neoantigen,length = len(df_ret))
    else:
        print('STMHCPan predicting......')
        df_ret = presentation_compute_sa(peptide, mhc, is_neoantigen, is_save)
        return render_template('result.html',peptide=peptide,mhc=mhc,score=0,p=df_ret['peptide'].to_list(),m=df_ret['allele'].to_list(),i=df_ret['neoantigen'].to_list(),binding=0,b=df_ret['presentation'].to_list(),is_save=is_save, length = len(df_ret))

    


@app.route('/download')
def download():
    return send_file("download/result.csv",as_attachment=True, cache_timeout=0)


@app.route('/learn_more')
def learn_more():
    return render_template('learn_more.html')




@app.route('/',methods=['GET','POST'])
def home():
    form = QueryForm()
    if request.method=='POST':
        session['peptide'] = form.peptide.data
        session['mhc'] = form.mhc.data
        session['neoantigen'] = form.neoantigen.data  # boolean
        session['save'] = form.save.data # boolean
        session['ma'] = form.ma.data # boolean
        if check_peptide(session.get('peptide')) and check_mhc(session.get('mhc')) and form.file_upload.data is None:
            '''
            this condition means the users have valid peptide and mhc input, and no file uploaded
            '''
            return redirect(url_for('result',peptide=session.get('peptide'),mhc=session.get('mhc'),is_neoantigen=session.get('neoantigen'),is_save=session.get('save'),is_ma=session.get('ma')))

        elif (form.file_upload.data is None and check_peptide(session.get('peptide'))==False) or (form.file_upload.data is None and check_mhc(session.get('mhc'))==False):
            '''
            this condition means the users have either invalid peptide or mhc input, and no file uploaded
            '''
            flash("Please check your peptide and MHC input!")
            return redirect(url_for('home'))


        else:
            '''
            this condition means the users have file uploaded
            '''
            print(session.get('peptide'))
            print(session.get('mhc'))
            print(session.get('ma'))
            if check_peptide(session.get('peptide'))==False and check_mhc(session.get('mhc'))==False:
                '''
                no peptide and mhc input
                '''
                uploaded_file = form.file_upload.data  # either a filestorage object or NoneType
                if True:
                    uploaded_file.save("./uploaded/multiple_query.csv")
                    is_neoantigen = session.get('neoantigen')
                    is_save = session.get('save')
                    is_ma = session.get('ma')
                    # is_right = check_upload('./uploaded/multiple_query.txt')
                    is_right = True
                    # print(is_right)
                    df_ret = file_process(is_ma)
                    print(df_ret)
                    if False:
                        flash("please check your input format:")
                        return redirect(url_for('home'))
                    else:
                        return redirect(url_for('download'))
                else:  # the file > certain size limit
                    flash("Currently we only support file less than 5MB")
                    return redirect(url_for('home'))
            else:
                '''
                have peptide or mhc input
                '''
                flash("You have uploaded files and inputted peptide or mhc")
                flash("please do either single query or multiple query")
                return redirect(url_for('home'))
    else:
        return render_template('submit.html',form=form)   










