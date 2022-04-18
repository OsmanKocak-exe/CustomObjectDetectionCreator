try:
    import tkinter.messagebox as tkm
    import cv2 
    import numpy as np  
    import shutil,os
    import tensorflow as tf
    import glob
    from object_detection.utils import label_map_util
    from object_detection.utils import visualization_utils as viz_utils
    from tkinter import ttk, filedialog, IntVar, Toplevel, Entry
    from ttkthemes import ThemedTk
    from object_detection.protos import pipeline_pb2
    from google.protobuf import text_format
    from object_detection.builders import model_builder
    from object_detection.utils  import config_util
except Exception as e:
         tkm.showinfo('ERROR:',str(e))
         
PYPATH = os.path.dirname(__file__)
print(PYPATH)
MAINPATH = 'Tensorflow'
LABELIMGPYPATH = MAINPATH+'/labelImg-master/labelImg.py'
WORKSPACE_PATH = MAINPATH+'/workspace'
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/models'
ANNOTATION_PATH = WORKSPACE_PATH +'/annotations'
IMAGE_PATH = WORKSPACE_PATH+'/images'
MODEL_PATH = WORKSPACE_PATH+'/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'
CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
CONFIG_PATH = MODEL_PATH+'/'+CUSTOM_MODEL_NAME+'/pipeline.config'

mainW = ThemedTk(theme="Arc", themebg=True, toplevel=True)
mainW.title("Create Custom Object Detaction Model")
mainWFX = 452
mainWFY = 535
mainW.geometry(str(mainWFX)+"x"+str(mainWFY))


#globalvars
createdfolderNameList = []
createdfolderPathList = []

num_train_steps = IntVar()
num_batch_size = IntVar()

getFolderBool = False
#

class tkinterGUI:
    
    try:
        ttkstyle = ttk.Style()
        ttkstyle.configure('Red.TLabelframe.Label',
                    font=("TkDefaultFont", 8, "bold"))

        btnctgory = ttk.Button(mainW, text="Choose Categories",
                            command=lambda: tkinterGUI.chsCatg(mainW))
        btnctgory.place(x=1, y=15, height=75, width=451)
        ttk.Label(btnctgory, text="Step 1", style='Red.TLabelframe.Label', foreground='gray').place(x=205,y=5)
        btnslcimg = ttk.Button(mainW, text="Select Images",
                            command=lambda: systemFunctions.selectImgs())
        btnslcimg.place(x=1, y=106, height=75, width=451)
        ttk.Label(btnslcimg, text="Step 2", style='Red.TLabelframe.Label', foreground='gray').place(x=205,y=5)
        btnsplit = ttk.Button(mainW, text="Train Test split",
                            command=lambda: systemFunctions.traintestIMGSplit())
        btnsplit.place(x=1, y=197, height=75, width=451)
        ttk.Label(btnsplit, text="Step 3", style='Red.TLabelframe.Label', foreground='gray').place(x=205,y=5)
        btntlabel = ttk.Button(mainW, text="Train the Model",
                            command=lambda: systemFunctions.trainlabel())
        btntlabel.place(x=1, y=288, height=75, width=451)
        ttk.Label(btntlabel, text="Step 4", style='Red.TLabelframe.Label', foreground='gray').place(x=205,y=5)

        btndetectOnCam = ttk.Button(mainW, text="WebCam Object Detection", command=lambda: systemFunctions.captureCam(0))
        btndetectOnCam.place(x=1, y=379, height=37, width=451)

        btndetectVideo = ttk.Button(mainW, text="Video Object Detection", command=lambda: systemFunctions.captureCam(1))
        btndetectVideo.place(x=1, y=417, height=37, width=451)

        ttk.Label(mainW, text="Step 5", style='Red.TLabelframe.Label', foreground='gray').place(x=205,y=410)
        ttk.Label(mainW, text="osmankocakank@gmail.com", style='Red.TLabelframe.Label', foreground='gray').place(x=150,y=520)
        btntsettings = ttk.Button(mainW, text="Settings",
                            command=lambda: tkinterGUI.settingsW(mainW))
        btntsettings.place(x=1, y=460, height=30, width=225)
        btntexit = ttk.Button(mainW, text="Exit",
                            command=lambda: mainW.destroy())
        btntexit.place(x=226, y=460, height=30, width=225)    

        btntboard = ttk.Button(mainW, text="Tensor Board",
                            command=lambda: systemFunctions.TensorBoard())
        btntboard.place(x=1, y=491, height=30, width=451)        
    except Exception as e:
         tkm.showinfo('ERROR:',str(e))
    def chsCatg(mainW):
        try:
            global ctgW
            ctgW = Toplevel(mainW)
            ctgW.title("Object Detection Model Categories")
            ctgW.geometry("261x500")
            ttk.Label(ctgW, text="Enter Category name(*): (Exp: Smoking)", foreground="grey").place(x=5,y=1)
            ttk.Label(ctgW, text="Enter Category name : (Exp: Not_Smoking)", foreground="grey").place(x=5,y=70)
            ttk.Label(ctgW, text="Enter Category name :", foreground="grey").place(x=5,y=141)
            ttk.Label(ctgW, text="Enter Category name :", foreground="grey").place(x=5,y=211)
            ttk.Label(ctgW, text="Enter Category name :", foreground="grey").place(x=5,y=281)
            ttk.Label(ctgW, text="Enter Category name :", foreground="grey").place(x=5,y=351)
            
            textbox0 = Entry(ctgW)
            textbox0.place(x=5,y=21,height=25, width=250)
            textbox0.focus()
            textbox1 = Entry(ctgW)
            textbox1.place(x=5,y=91,height=25, width=250)
            textbox2 = Entry(ctgW)
            textbox2.place(x=5,y=161,height=25, width=250)
            textbox3 = Entry(ctgW)
            textbox3.place(x=5,y=231,height=25, width=250)
            textbox4 = Entry(ctgW)
            textbox4.place(x=5,y=301,height=25, width=250)
            textbox5 = Entry(ctgW)
            textbox5.place(x=5,y=371,height=25, width=250)
            btnsave = ttk.Button(ctgW, text="SAVE",
                        command=lambda:systemFunctions.saveCatg(textbox0,textbox1,textbox2,textbox3,textbox4,textbox5))
            btnsave.place(x=5, y=421, height=50, width=250)
        except Exception as e:
         tkm.showinfo('ERROR:',str(e))

    def settingsW(mainW):
        try:
            stngW = Toplevel(mainW)
            stngW.title("Settings")
            stngW.geometry("251x150")
            ttk.Label(stngW, text="Choose Train Steps :", foreground="grey").place(x=1,y=1)
            ttk.Label(stngW, text="Choose Batch Size :", foreground="grey").place(x=1,y=50)
            
            trainSCmb = ttk.Combobox(stngW, textvariable=num_train_steps)
            trainSCmb.place(x=2,y=20,height=25,width=100)
            trainSCmb['values'] = (1000, 3000, 5000, 8000, 10000, 15000)
            trainSCmb['state'] = 'readonly'

            numBSCmb = ttk.Combobox(stngW, textvariable=num_batch_size)
            numBSCmb.place(x=2,y=74,height=25,width=100)
            numBSCmb['values'] = (5, 15, 25, 50, 100, 200, 350, 500)
            numBSCmb['state'] = 'readonly'

            btnsave = ttk.Button(stngW, text="SAVE",
                        command=lambda: stngW.destroy())
            btnsave.place(x=120, y=1, height=100, width=131)

            btnsetdef = ttk.Button(stngW, text="Reset to Default",command=lambda:systemFunctions.returnDefSett())
            btnsetdef.place(x=1, y=100, height=50, width=250)
        except Exception as e:
         tkm.showinfo('ERROR:',str(e))


class PiplelineConfig:
    def pipevars(r):
        try:
            pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
            with tf.io.gfile.GFile(CONFIG_PATH, "r") as f:                                                                                                                                                                                                                     
                proto_str = f.read()                                                                                                                                                                                                                                          
                text_format.Merge(proto_str, pipeline_config) 
            #pipevars
            pipeline_config.model.ssd.num_classes = r+1
            pipeline_config.train_config.batch_size = num_batch_size.get()
            pipeline_config.train_config.fine_tune_checkpoint = PRETRAINED_MODEL_PATH+'/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0'
            pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
            pipeline_config.train_input_reader.label_map_path= ANNOTATION_PATH + '/label_map.pbtxt'
            pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/train.record']
            pipeline_config.eval_input_reader[0].label_map_path = ANNOTATION_PATH + '/label_map.pbtxt'
            pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/test.record']
            #
            config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
            with tf.io.gfile.GFile(CONFIG_PATH, "wb") as f:                                                                                                                                                                                                                     
                f.write(config_text)   
        except Exception as e:
         tkm.showinfo('ERROR:',str(e))

class systemFunctions:
    def returnDefSett():
        num_train_steps.set(5000)
        num_batch_size.set(5) 

    def saveCatg(textbox0,textbox1,textbox2,textbox3,textbox4,textbox5):
            inputValue=textbox0.get(),textbox1.get(),textbox2.get(),textbox3.get(),textbox4.get(),textbox5.get()

            lblerr = ttk.Label(ctgW, text="", foreground="red")
            lblerr.place(x=5,y=400)
            lblerr.config(text='')
            try:
                for i in range(len(inputValue)):     
                    if (inputValue[i] != ''):
                        newfoldername = inputValue[i]
                        newfloderpath = IMAGE_PATH +'/'+ newfoldername
                        createdfolderNameList.append(inputValue[i])
                        createdfolderPathList.append(newfloderpath)
                        if (not os.path.exists(newfloderpath)):
                            os.makedirs(newfloderpath)
                            lblerr.config(text='Folder(s) successfully created', foreground='green')
                        else:
                            lblerr.config(text='Folder Name: '+'" '+newfoldername+' "'+' already exists')
                            print('Folder Name: '+newfoldername+'already exists')
                    elif(inputValue[0] == ''):
                        lblerr.config(text='Please fill the required areas (*)')
                        print('Please fill the required areas (*)')
                        break
            except Exception as e:
                print('Error: ',e)


    def getFolderNames():
        global getFolderBool
        getFolderBool = True
        def selectFolders():
            try:
                open_folder_path = filedialog.askdirectory(title='Select Folders from -> \objdetection\Tensorflow\workspace\images')
                foldername = os.path.basename(open_folder_path)
                
                createdfolderNameList.append(foldername)
                createdfolderPathList.append(open_folder_path)
                msgBoxforselect()
            except Exception as e:
                tkm.showinfo('ERROR:',str(e))
        def msgBoxforselect():
            try:
                if(len(createdfolderNameList) == 0 or createdfolderNameList[0] == ''):
                    print('Canceled')
                else:
                    MsgBox = tkm.askquestion ('Message','Do you want select more? (Folders must be empty!)',icon = 'warning')
                    if MsgBox == 'yes':
                        selectFolders()
                    else:
                        tkm.showinfo('Message','Selected folders are -> {} press OK for continue to Step-3'.format(str(createdfolderNameList)))
            except Exception as e:
                tkm.showinfo('ERROR:',str(e))
        tkm.showinfo("Missing step: Unselected category.", "Hi, We can't detect any 'Created Category Folder', if you have created folder before, select one by one, or return to Step-1")
        selectFolders()
        
        
    def selectImgs():
        
        if(not os.path.exists(IMAGE_PATH+'/beckupimages')):
            os.makedirs(IMAGE_PATH+'/beckupimages')
        if(len(createdfolderNameList) == 0):
            systemFunctions.getFolderNames()
        try:
            for x in range(len(createdfolderNameList)):
                if(createdfolderNameList[0] == ''):
                    tkm.showinfo('ERROR:','Unselected category')
                    break
                srcimgfiles = filedialog.askopenfilenames(parent=mainW, title='Choose images files for ' +' "' + createdfolderNameList[x] +'" ' , filetypes=(('image .jpg', 'image .png')))
                try:
                    for i in range(len(srcimgfiles)):
                        filepath = os.path.splitext(srcimgfiles[i])[0]     
                        extension = os.path.splitext(srcimgfiles[i])[1]
                        flename = os.path.basename(srcimgfiles[i])
                        flename = flename.replace(extension, '')
                        filepath = filepath[:(len(filepath)-len(flename))]
                        print(flename)
                        shutil.copy(srcimgfiles[i], IMAGE_PATH+'/beckupimages')
                        #filepath = filepath.replace(flename, '')
                        print(createdfolderPathList[x],createdfolderNameList[x])
                        if(extension == '.jpg'):
                            os.rename(srcimgfiles[i], filepath+(str(i))+'_'+createdfolderNameList[x]+'.jpg')
                            if(getFolderBool == True):
                                shutil.move(filepath+str(i)+'_'+createdfolderNameList[x]+extension, createdfolderPathList[x]+'/')
                            else:
                                shutil.move(filepath+str(i)+'_'+createdfolderNameList[x]+extension, PYPATH+'/'+createdfolderPathList[x]+'/')
                        else:
                            os.rename(srcimgfiles[i], filepath+(str(i))+'_'+createdfolderNameList[x]+'.png')
                            if(getFolderBool == True):
                                shutil.move(filepath+str(i)+'_'+createdfolderNameList[x]+extension, createdfolderPathList[x]+'/')
                            else:
                                shutil.move(filepath+str(i)+'_'+createdfolderNameList[x]+extension, PYPATH+'/'+createdfolderPathList[x]+'/')
                except Exception as e:
                    print('Error: selectimgs for no 2',e)
                    tkm.showinfo('ERROR:',e)
            if(not createdfolderNameList[0] == '' and not createdfolderNameList == None and not srcimgfiles == ''):
                os.system("cmd /c python {}".format(LABELIMGPYPATH))
            else:
                tkm.showinfo('ERROR:','Images not selected!')
        except Exception as e:
            print('Error: selectimgs for no 1',e)
            tkm.showinfo('ERROR:',e)
        

    def traintestIMGSplit():

        try:
            for x in range(len(createdfolderNameList)):
                srcsplitimgfiles = filedialog.askopenfilenames(parent=mainW, title='Choose images files for split from folder: ' +' "' + createdfolderNameList[x] +'" ' , filetypes=(('image .jpg', 'image .png')))  
                try:
                    for i in range(len(srcsplitimgfiles)):
                        if(i <= int((len(srcsplitimgfiles)*60)/100)):
                            print ('trainn',i,srcsplitimgfiles[i])
                            shutil.move(srcsplitimgfiles[i], IMAGE_PATH+'/train')
                        else:
                            print ('test',i,srcsplitimgfiles[i])
                            shutil.move(srcsplitimgfiles[i], IMAGE_PATH+'/test')
                except Exception as e:
                    print('Error: traintestImgsplit for no 2',e)
                    tkm.showinfo('ERROR:',e)
            systemFunctions.trantestXMLSplit()
        except Exception as e:
            print('Error: traintestImgsplit for no 1',e)
            tkm.showinfo('ERROR:',e)
        

    def trantestXMLSplit():
        
        try:
            for x in range(len(createdfolderNameList)):
                srcsplitxmlfiles = filedialog.askopenfilenames(parent=mainW, title='Choose XML files for split from folder: ' +' "' + createdfolderNameList[x] +'" ' , filetypes=(('xml .xml','xml ""')))  
                try:
                    for i in range(len(srcsplitxmlfiles)):
                        if(i <= int((len(srcsplitxmlfiles)*60)/100)):
                            print ('trainn',i,srcsplitxmlfiles[i])
                            shutil.move(srcsplitxmlfiles[i], IMAGE_PATH+'/train')
                        else:
                            print ('test',i,srcsplitxmlfiles[i])
                            shutil.move(srcsplitxmlfiles[i], IMAGE_PATH+'/test')
                except Exception as e:
                    print('Error: traintestXmlsplit no 2',e)
                    tkm.showinfo('ERROR:',e)
            systemFunctions.createLabelmap()
        except Exception as e:
            print('Error: traintestXmlsplit for no 1',e)
            tkm.showinfo('ERROR:',e)

    def createLabelmap():
        try:
            labels = []
            for r in range(len(createdfolderNameList)):
                    labels.append({'name':str(createdfolderNameList[r]),'id':r})
        
            print(type(labels),labels)
            if(not r == 0):
                with open(ANNOTATION_PATH + '/label_map.pbtxt','w') as f:
                    for label in labels:    
                        f.write('item{\n')
                        f.write('\tname:\'{}\'\n'.format(label['name']))
                        f.write('\tid:{}\n'.format(label['id']+1))
                        f.write('}\n')
                PiplelineConfig.pipevars(r)
                systemFunctions.tfRecord()
        except Exception as e:
            tkm.showinfo('ERROR:',str(e))
        
    def tfRecord():
        try:
            tfrsTrain = "python {}/generate_tfrecord.py -x {}/train -l {}/label_map.pbtxt -o {}/train.record".format(SCRIPTS_PATH,IMAGE_PATH,ANNOTATION_PATH,ANNOTATION_PATH)
            tfrsTest = "python {}/generate_tfrecord.py -x {}/test -l {}/label_map.pbtxt -o {}/test.record".format(SCRIPTS_PATH,IMAGE_PATH,ANNOTATION_PATH,ANNOTATION_PATH)

            os.system("start /wait cmd /c {}".format(tfrsTrain))
            os.system("start /wait cmd /c {}".format(tfrsTest))
        except Exception as e:
            tkm.showinfo('ERROR:',str(e))


    def ctgrIndex():
        try:
            category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')
            print(category_index)
            return category_index
        except Exception as e:
            tkm.showinfo('ERROR:',str(e))
    def trainlabel():
        try:
            print("""python {}/research/object_detection/model_main_tf2.py  --model_dir={}/{}  --pipeline_config_path={}/{}/pipeline.config --num_train_steps={}"""
            .format(APIMODEL_PATH, MODEL_PATH,CUSTOM_MODEL_NAME,MODEL_PATH,CUSTOM_MODEL_NAME,num_train_steps.get()))

            trainstring ='python {}/research/object_detection/model_main_tf2.py --model_dir={}/{} --pipeline_config_path={}/{}/pipeline.config --num_train_steps={}'.format(
                APIMODEL_PATH, MODEL_PATH,CUSTOM_MODEL_NAME,MODEL_PATH,CUSTOM_MODEL_NAME,num_train_steps.get())
            os.system("start /wait cmd /c {}".format(trainstring))
        except Exception as e:
            tkm.showinfo('ERROR:',str(e))

    def TensorBoard():
        try:
            boardstring = 'tensorboard --logdir {}/{}/{}'.format(PYPATH,MODEL_PATH,CUSTOM_MODEL_NAME)
            os.system("start /wait cmd /c {}".format(boardstring))
        except Exception as e:
            tkm.showinfo('ERROR:',str(e))

    def detection_models(lastedckptname):
        try:
            configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
            detection_model = model_builder.build(model_config=configs['model'], is_training=False)


            ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
            ckpt.restore(os.path.join(CHECKPOINT_PATH, lastedckptname)).expect_partial()
            return detection_model
        except Exception as e:
            tkm.showinfo('ERROR:',str(e))

    def detect_fn(image,detection_model): 
        try:
            image, shapes = detection_model.preprocess(image)
            prediction_dict = detection_model.predict(image, shapes)
            detections = detection_model.postprocess(prediction_dict, shapes)
            return detections
        except Exception as e:
            tkm.showinfo('ERROR:',str(e))

    def findLastedCheckpoint():
        try:
            filenames = glob.glob(PYPATH+'/'+CHECKPOINT_PATH+'*'+'.index')

            for i in range(len(filenames)):
                extension = os.path.splitext(filenames[i])[1]
                flename = os.path.basename(filenames[i])
                flename = flename.replace(extension, '')    
            if(flename == '' or flename == None):
                tkm.showinfo('ERROR:',' Cannot found Checkpoint.')
            elif(flename == 'ckpt-0'):
                tkm.showinfo('ERROR:',' Please Train the model first.')
                return flename
            else:
                return flename
        except Exception as e:
            tkm.showinfo('ERROR:',str(e) + ' Checkpoint files is missing or empty.')

    def captureCam(buttonindex):
        lastedckptname = systemFunctions.findLastedCheckpoint()
        detection_model = systemFunctions.detection_models(lastedckptname)
        try:
            catIndex = systemFunctions.ctgrIndex()     
            if(buttonindex == 1):
                srcvideofile = filedialog.askopenfilenames(parent=mainW, title='Choose video file', filetypes=(('Video .mp4', 'Video .avi')))     
                cap = cv2.VideoCapture(str(srcvideofile[0]))
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                fps = cap.get(cv2.CAP_PROP_FPS)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                video = cv2.VideoWriter(PYPATH+'/'+MAINPATH+'/'+'exportedvideos/'+'expVideo.mp4', fourcc, fps, (int(width), int(height)))

                while True: 
                    ret, frame = cap.read()
                    image_np = np.array(frame)
                    if (image_np.size == 1):
                        tkm.showinfo('INFO: Video Exported Successfully', ' Video exported to '+ PYPATH+'/'+MAINPATH+'/'+'exportedvideos/'+'expVideo.mp4')
                        break
                    else:
                        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
                        detections = systemFunctions.detect_fn(input_tensor,detection_model)
                        
                        num_detections = int(detections.pop('num_detections'))
                        detections = {key: value[0, :num_detections].numpy()
                                    for key, value in detections.items()}
                        detections['num_detections'] = num_detections

                        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

                        label_id_offset = 1
                        image_np_with_detections = image_np.copy()

                        viz_utils.visualize_boxes_and_labels_on_image_array(
                                    image_np_with_detections,
                                    detections['detection_boxes'],
                                    detections['detection_classes']+label_id_offset,
                                    detections['detection_scores'],
                                    catIndex,
                                    use_normalized_coordinates=True,
                                    max_boxes_to_draw=5,
                                    min_score_thresh=.5,
                                    agnostic_mode=False)                   
                        cv2.imshow('Video Exporting.. press "q" for end',  cv2.resize(image_np_with_detections, (int(width), int(height))))
                        video.write(image_np_with_detections)      
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            cap.release()
                            break               
            else:
                cap = cv2.VideoCapture(0)
                while True: 
                    ret, frame = cap.read()
                    image_np = np.array(frame)
                    
                    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
                    detections = systemFunctions.detect_fn(input_tensor,detection_model)
                    
                    num_detections = int(detections.pop('num_detections'))
                    detections = {key: value[0, :num_detections].numpy()
                                for key, value in detections.items()}
                    detections['num_detections'] = num_detections

                    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

                    label_id_offset = 1
                    image_np_with_detections = image_np.copy()

                    viz_utils.visualize_boxes_and_labels_on_image_array(
                                image_np_with_detections,
                                detections['detection_boxes'],
                                detections['detection_classes']+label_id_offset,
                                detections['detection_scores'],
                                catIndex,
                                use_normalized_coordinates=True,
                                max_boxes_to_draw=5,
                                min_score_thresh=.5,
                                agnostic_mode=False)     
                    cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))
                        
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cap.release()
                        break
        except Exception as e:
            tkm.showinfo('ERROR:',str(e) + ' Plase make sure your webcam is working properly.')

systemFunctions.returnDefSett()
mainW.mainloop()