import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def preprocessing():
    print("Mulai proses otomatisasi preprocessing...")

    ## Load dataset

    # Mencari lokasi absolut folder tempat script ini berada (folder 'preprocessing')
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Menggabungkan path script dengan lokasi dataset
    input_path = os.path.join(script_dir, "..", "college_student_placement_dataset_raw.csv")
    
    # Menyimpan path output ditempat script ini berada
    output_path = os.path.join(script_dir, "college_student_placement_dataset_preprocessing.csv")

    # Memuat dataset
    try:
        df = pd.read_csv(input_path)
        print(f"Dataset berhasil dimuat dari: {input_path}")
        print(f"Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di: {input_path}")
        return

    ## Preprocessing
    
    # Drop College_ID
    if 'College_ID' in df.columns:
        df = df.drop(columns=['College_ID'])

    # Drop kolom duplikat    
    df = df.drop_duplicates()

    # Encoding data kategorikal
    mapping = {'Yes': 1, 'No': 0}
    
    # Cek kolom sebelum mapping untuk menghindari error jika dijalankan ulang
    if 'Internship_Experience' in df.columns and df['Internship_Experience'].dtype == 'object':
        df['Internship_Experience'] = df['Internship_Experience'].map(mapping)
        
    if 'Placement' in df.columns and df['Placement'].dtype == 'object':
        df['Placement'] = df['Placement'].map(mapping)

    # Inisialisasi Scaler
    scaler = StandardScaler()
    
    fitur_numerik = [
        "IQ", 'Prev_Sem_Result', "CGPA", 'Academic_Performance', 
        "Extra_Curricular_Score", "Communication_Skills", "Projects_Completed"
    ]

    # Standarisasi fitur numerik
    df[fitur_numerik] = scaler.fit_transform(df[fitur_numerik])

    # Menyimpan data bersih
    df.to_csv(output_path, index=False)
    
    print(f"Preprocessing Selesai! Data bersih tersimpan di: {output_path}")

if __name__ == "__main__":
    preprocessing()