################################################
Written by: Omar Omari - Friday 21st May, 2021
Title: ReadMe for DSP_Part1.py
################################################

This is the 1st of 3 steps to running DSP Part 1, also known as “The data-pipeline process”. This ReadMe file deals only with setting up and running “DSP_Part1.py”. The structure of this document is as follows:

1.) Purpose of overall step
2.) Things to do before opening the main script
3.) Things to do after opening the main script
4.) Running the main script

----------------------------------------------------------------------------------------------------------------------------
1.) Purpose:
To generate the RF, Quantized. Non-scanconverted B-mode, QUS, NPS and PS data on both the Video and Frame-level, along with their associated dataframes on the Case, Video and Frame-level.

2.) Things to do before opening the main script:
- In the GCP Storage area, create a new storage bucket for where your processed cases will be stored along with their dataframes.

- Log into “data-processing” instance and go the following directory:
liver_dl_pipeline/liver_project_storage_structure/universal_datagen/

Note, this will be the main directory for where all operations for this stage in the data processing take place

- Prepare “PD.csv” file from the downloaded .csv file from the cases dashboard, and place PD.csv in same directory as “DSP_Part1.py”.

- Create a new local directory entitled “Storage_Bucket_name”_dataframes_”TodaysDate” in the same directory above.

- This is already done, BUT, ensure a directory called “temp_files” exists in the same directory above.

3.) Things to do after opening the main script:
-Open “DSP_Part1-NoFrameLevelPD_cleaned.py” and scroll down to where your reach the following lines:

###################################################################################
# THE STUFF THAT NEEDS TO RUN

print('STARTING PREP')
###################################################################################

-  Specify bucket where data/cases will be pulled fro to be processed in list “bucket_list”
	Note, if processing new cases, this almost always will be "pd-cases-tar-extracted"

- Specify bucket where processed cases will be pushed in the list “bucket_push_name”
	Note, this will be name of the bucket you created in the GCP Storage.

- Specify the CSV file for where the cases to be processed will be loaded in the list “cases_list”. This is just the “PD.csv” file.

- Redefine or keep window-size dimensions “depth” and “width”, and the length of the proximal and distal windows used in AttenuationCoeff2 “freq_num_for_atten”.
	Note, these will likely not change, but this should be the default values
    depth=122
    width=4
    freq_num_for_att=25

- Scroll down to where you start the PD process,

###############################################
print('Finished DSP, now starting PD process')
###############################################

And then down to where you are fetching the PD dataframe,

###############################################
print("Fetching the patient demographic data dataframe")
PD_DF1 = pd.read_csv('PD_Nov22_2021_df.csv')
###############################################

Insert “PD.csv” file name in pd.read_csv to get patient demographic info.

Note, when you receive the PD.csv file from Miriam, be sure that the column headers match the columns listed in list “wanted_data”, [ 'Disease' , 'History' , 'BMI' , 'FibroscanCAP' , 'FibroscanKPA' , 'Country' , 'Gender' , 'Age' , 'number_of_lesions', 'lesion_size', 'Notes'].

OR more recently, the columns in-question should be: ['Patient_id', 'Repeated?', 'Site', 'Height', 'Weight',
   'BMI', 'Age', 'Gender', 'History1', 'History2', 'History3',
   'History hbv', 'History hcv', 'History (yes/no)', 'KPA',
   'Fibrosis Grade', 'CAP', 'Steatosis Grade', 'Disease Labelled',
   'Disease', 'Unlabelled Disease', 'Payable', 'irb', 'Ascites',
   'Portal Vein Thrombosis', 'Biopsy', 'Jaundice']

- Scroll down to the comment “ Save and upload all processed cases”, and update the “..._dataframes_.../case_data.pkl”, “..._dataframes_.../video_data.pkl”, “..._dataframes_.../frame_data.pkl” directories to your newly created directory “Storage_Bucket_name”_dataframes_”TodaysDate”.

4.) Running the main script:
- Open a command terminal and change-directory to “liver_dl_pipeline/liver_project_storage_structure/universal_datagen/”.

- Run the “DSP_Part1-NoFrameLevelPD_cleaned.py” script using a nohup command via,

nohup python -u ./DSP_Part1-NoFrameLevelPD_cleaned.py > logfile_to_store_and_monitor_output_from_running_script.log &

Note, change the .log name to something that you would recognize. Best practice would be the date the script is being run, the batch # that is being processed, e.g. US_cases_Mar2021.
----------------------------------------------------------------------------------------------------------------------------

This process may take a few hours, maybe a day depending on the size and number of cases being processed. The nohup command will allow you to be able to close your browser and have your instance process the cases, just be sure to leave your instance running for this to work. Periodically check the log file to monitor for any errors, and maybe also run an htop command in another terminal to monitor the CPU activity as well.

This concludes Part 1 of 3 of the data-pipeline process.
