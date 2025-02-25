import streamlit as st
import os
import gdown
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from scipy.spatial import cKDTree
import gspread
from google.oauth2 import service_account
from concurrent.futures import ThreadPoolExecutor
import re
import glob

# Configuración de la página
st.set_page_config(
    page_title="Herramientas de Geolocalización",
    page_icon="🌍",
    layout="wide"
)

# Configuración de estilos
st.markdown("""
    <style>
    .stAlert {
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Enlaces de los archivos shape
file_urls = {
    "combined.shp": "https://drive.google.com/uc?id=1UBuvlHzBGV4CNArXTJsOUeu4kW-M4V1Z",
    "combined.dbf": "https://drive.google.com/uc?id=1sLEdQDHX3yKof2q7wjhH3R-yqACdit0g",
    "combined.prj": "https://drive.google.com/uc?id=1f7R7N6gs9en-RJr3vY21Gxjxesal6ZQg",
    "combined.shx": "https://drive.google.com/uc?id=1MfeWFsxlA7EUR3sxnC3WC--ukevrMei8"
}

# --- 1. Funciones de carga de archivos ---
@st.cache_resource
def download_shapefiles():
    """Descarga los archivos shape y retorna la ruta del directorio"""
    download_folder = os.path.join(os.getcwd(), "shapefile_downloaded")
    os.makedirs(download_folder, exist_ok=True)
    
    # Verificar si archivos ya existen y eliminarlos para evitar problemas
    for filename in os.listdir(download_folder):
        if filename.startswith("combined."):
            os.remove(os.path.join(download_folder, filename))
    
    # Descargar los archivos
    for filename, url in file_urls.items():
        destination = os.path.join(download_folder, filename)
        try:
            gdown.download(url, destination, quiet=False)
            # Verificar que el archivo se descargó correctamente
            if not os.path.exists(destination) or os.path.getsize(destination) == 0:
                raise Exception(f"Error descargando {filename}")
        except Exception as e:
            st.error(f"Error al descargar {filename}: {str(e)}")
            raise e
    
    return download_folder

@st.cache_resource
def load_shapefile(shapefile_path):
    """Carga el archivo shape"""
    try:
        # Verificar que el archivo existe
        if not os.path.exists(shapefile_path):
            files = glob.glob(os.path.join(os.path.dirname(shapefile_path), "*"))
            st.error(f"El archivo {shapefile_path} no existe. Archivos disponibles: {files}")
            raise FileNotFoundError(f"El archivo {shapefile_path} no existe")
        
        # Verificar que todos los archivos necesarios existen
        base_path = os.path.splitext(shapefile_path)[0]
        required_extensions = ['.shp', '.dbf', '.prj', '.shx']
        for ext in required_extensions:
            file_path = f"{base_path}{ext}"
            if not os.path.exists(file_path):
                st.error(f"Archivo requerido {file_path} no encontrado")
                raise FileNotFoundError(f"Archivo requerido {file_path} no encontrado")
        
        # Cargar el archivo shapefile
        return gpd.read_file(shapefile_path)
    except Exception as e:
        st.error(f"Error al cargar el shapefile: {str(e)}")
        raise e

def vectorized_parse_coordinates(series):
    """Procesa las coordenadas de forma vectorizada"""
    series = pd.to_numeric(series.astype(str).str.replace(',', '.'), errors='coerce')
    mask = np.abs(series) > 180
    series[mask] = series[mask] / 100000000.0
    return series

# --- 2. Funciones de Conexión y Carga de Datos ---
def init_connection():
    """Función para inicializar la conexión con Google Sheets."""
    try:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=[
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]
        )
        client = gspread.authorize(credentials)
        return client
    except Exception as e:
        st.error(f"Error en la conexión: {str(e)}")
        return None

def load_sheet(client):
    """Función para cargar la hoja de trabajo de Google Sheets."""
    try:
        return client.open_by_url(st.secrets["spreadsheet_url"]).sheet1
    except Exception as e:
        st.error(f"Error al cargar la planilla: {str(e)}")
        return None

# --- 3. Funciones para Georreferenciación ---
def process_data(data, gdf):
    """Procesa los datos y realiza la georreferenciación"""
    try:
        # Verificar que existen las columnas necesarias
        required_columns = ['Latitud campo', 'Longitud Campo']
        for col in required_columns:
            if col not in data.columns:
                st.error(f"Columna requerida '{col}' no encontrada en los datos")
                raise ValueError(f"Columna requerida '{col}' no encontrada en los datos")
        
        # Procesar coordenadas
        data['lat'] = vectorized_parse_coordinates(data['Latitud campo'])
        data['lon'] = vectorized_parse_coordinates(data['Longitud Campo'])
        
        # Identificar coordenadas inválidas
        invalid_mask = data['lat'].isna() | data['lon'].isna() | \
                    np.isinf(data['lat']) | np.isinf(data['lon'])
        valid_data = data[~invalid_mask].copy()
        
        # Si no hay datos válidos, retornar resultados vacíos
        if len(valid_data) == 0:
            st.warning("No se encontraron coordenadas válidas para procesar")
            empty_results = pd.DataFrame(columns=['Region', 'Provincia', 'Comuna', 'original_index'])
            return empty_results, 0, len(data)
        
        # Crear array de centroides
        centroids = np.array([[geom.centroid.x, geom.centroid.y] for geom in gdf.geometry])
        tree = cKDTree(centroids)
        
        # Crear array de puntos válidos
        points = np.column_stack([valid_data['lon'].values, valid_data['lat'].values])
        
        # Encontrar el polígono más cercano
        distances, indices = tree.query(points, k=1)
        
        # Determinar puntos fuera del territorio
        DISTANCE_THRESHOLD = 1
        outside_mask = distances > DISTANCE_THRESHOLD
        
        # Crear DataFrame con resultados
        results = pd.DataFrame({
            'Region': gdf.iloc[indices]['Region'].values,
            'Provincia': gdf.iloc[indices]['Provincia'].values,
            'Comuna': gdf.iloc[indices]['Comuna'].values,
            'original_index': valid_data.index + 2
        })
        
        # Marcar puntos fuera como "OTROS"
        results.loc[outside_mask, ['Region', 'Provincia', 'Comuna']] = "OTROS"
        
        # Agregar filas con coordenadas inválidas
        na_results = pd.DataFrame({
            'Region': "NA",
            'Provincia': "NA",
            'Comuna': "NA",
            'original_index': data[invalid_mask].index + 2
        })
        
        final_results = pd.concat([results, na_results], ignore_index=True)
        return final_results, len(valid_data), len(data[invalid_mask])
    except Exception as e:
        st.error(f"Error al procesar datos: {str(e)}")
        raise e

def update_google_sheets_georreferencia(client, final_results):
    """Actualiza Google Sheets con los resultados de georreferenciación"""
    try:
        sheet = load_sheet(client)
        
        if len(final_results) == 0:
            st.warning("No hay resultados para actualizar en Google Sheets")
            return True
        
        # Preparar actualizaciones en lotes
        BATCH_SIZE = 100  # Reducido para evitar sobrecarga
        updates = []
        current_batch = []
        
        for _, row in final_results.iterrows():
            current_batch.append({
                'range': f'H{row.original_index}:J{row.original_index}',
                'values': [[row.Region, row.Provincia, row.Comuna]]
            })
            
            if len(current_batch) >= BATCH_SIZE:
                updates.append(current_batch)
                current_batch = []
        
        if current_batch:
            updates.append(current_batch)
        
        # Actualizar en lotes
        total_batches = len(updates)
        progress_bar = st.progress(0)
        
        for i, batch in enumerate(updates):
            sheet.batch_update(batch)
            progress_bar.progress((i + 1) / total_batches)
        
        return True
    except Exception as e:
        st.error(f"Error al actualizar Google Sheets: {str(e)}")
        return False

# --- 4. Funciones para aplicar formato a las celdas ---
def apply_format_sonda(sheet):
    """Aplica formato a las celdas de la hoja de cálculo para Sondas."""
    try:
        text_format = {
            "backgroundColor": {"red": 1, "green": 1, "blue": 1},
            "horizontalAlignment": "CENTER",
            "textFormat": {
                "foregroundColor": {"red": 0, "green": 0, "blue": 0},
                "fontFamily": "Arial",
                "fontSize": 11
            }
        }
        number_format = {
            "numberFormat": {
                "type": "NUMBER",
                "pattern": "#,##0.00000000"
            },
            "backgroundColor": {"red": 1, "green": 1, "blue": 1},
            "horizontalAlignment": "CENTER",
            "textFormat": {
                "foregroundColor": {"red": 0, "green": 0, "blue": 0},
                "fontFamily": "Arial",
                "fontSize": 11
            }
        }
        # Columna M: Ubicación sonda google maps (texto, DMS)
        sheet.format("M2:M", text_format)
        # Columnas N y O: Latitud sonda y Longitud sonda (números)
        sheet.format("N2:O", number_format)
    except Exception as e:
        st.warning(f"Error al aplicar formato: {str(e)}")

def apply_format_campo(sheet):
    """Aplica formato a las celdas de la hoja de cálculo para Campo."""
    try:
        text_format = {
            "backgroundColor": {"red": 1, "green": 1, "blue": 1},
            "horizontalAlignment": "CENTER",
            "textFormat": {
                "foregroundColor": {"red": 0, "green": 0, "blue": 0},
                "fontFamily": "Arial",
                "fontSize": 11
            }
        }
        number_format = {
            "numberFormat": {
                "type": "NUMBER",
                "pattern": "#,##0.00000000"
            },
            "backgroundColor": {"red": 1, "green": 1, "blue": 1},
            "horizontalAlignment": "CENTER",
            "textFormat": {
                "foregroundColor": {"red": 0, "green": 0, "blue": 0},
                "fontFamily": "Arial",
                "fontSize": 11
            }
        }
        # Columna E: Ubicación campo (texto, DMS)
        sheet.format("E2:E", text_format)
        # Columnas F y G: Latitud campo y Longitud Campo (números)
        sheet.format("F2:G", number_format)
    except Exception as e:
        st.warning(f"Error al aplicar formato: {str(e)}")

# --- 5. Función para formatear la cadena DMS ---
def format_dms(value):
    """Formatea una cadena DMS al formato correcto."""
    if not value or not isinstance(value, str):
        return None
    
    pattern = r'(\d+)[°º]\s*(\d+)[\']\s*([\d\.]+)"\s*([NS])\s+(\d+)[°º]\s*(\d+)[\']\s*([\d\.]+)"\s*([EW])'
    m = re.match(pattern, value.strip())
    if m:
        lat_deg, lat_min, lat_sec, lat_dir, lon_deg, lon_min, lon_sec, lon_dir = m.groups()
        try:
            lat_deg = int(lat_deg)
            lat_min = int(lat_min)
            lat_sec = float(lat_sec)
            lon_deg = int(lon_deg)
            lon_min = int(lon_min)
            lon_sec = float(lon_sec)
        except ValueError:
            return None
        formatted_lat = f"{lat_deg:02d}°{lat_min:02d}'{lat_sec:04.1f}\"{lat_dir.upper()}"
        formatted_lon = f"{lon_deg:02d}°{lon_min:02d}'{lon_sec:04.1f}\"{lon_dir.upper()}"
        return f"{formatted_lat} {formatted_lon}"
    return None

# --- 6. Actualizar el contenido de la columna DMS ---
def update_dms_format_column_sonda(sheet):
    """Actualiza la columna DMS en la hoja de cálculo para Sondas usando batch_update."""
    try:
        dms_values = sheet.col_values(13)  # Columna M
        if len(dms_values) <= 1:
            st.warning("No hay datos en la columna de ubicación de sonda")
            return
        
        start_row = 2
        batch_updates = []
        
        for i in range(start_row - 1, len(dms_values)):
            original_value = dms_values[i] if i < len(dms_values) else ""
            if original_value:
                new_val = format_dms(original_value)
                if new_val is not None:
                    batch_updates.append({"range": f"M{i+1}", "values": [[new_val]]})
        
        if batch_updates:
            sheet.batch_update(batch_updates)
    
    except Exception as e:
        st.warning(f"Error al actualizar formato DMS para sondas: {str(e)}")

def update_dms_format_column_campo(sheet):
    """Actualiza la columna DMS en la hoja de cálculo para Campo usando batch_update."""
    try:
        dms_values = sheet.col_values(5)  # Columna E
        if len(dms_values) <= 1:
            st.warning("No hay datos en la columna de ubicación de campo")
            return
        
        start_row = 2
        batch_updates = []

        for i in range(start_row - 1, len(dms_values)):
            original_value = dms_values[i] if i < len(dms_values) else ""
            if original_value:
                new_val = format_dms(original_value)
                if new_val is not None:
                    batch_updates.append({"range": f"E{i+1}", "values": [[new_val]]})

        if batch_updates:
            sheet.batch_update(batch_updates)

    except Exception as e:
        st.warning(f"Error al actualizar formato DMS para campo: {str(e)}")

# --- 7. Funciones de conversión ---
def dms_to_decimal(dms_str):
    """Convierte DMS a decimal."""
    if not dms_str or not isinstance(dms_str, str):
        return None
    
    pattern = r'(\d{1,2})[°º](\d{1,2})[\'](\d{1,2}(?:\.\d+)?)"([NS])\s+(\d{1,3})[°º](\d{1,2})[\'](\d{1,2}(?:\.\d+)?)"([EW])'
    m = re.match(pattern, dms_str.strip())
    if m:
        lat_deg, lat_min, lat_sec, lat_dir, lon_deg, lon_min, lon_sec, lon_dir = m.groups()
        try:
            lat = int(lat_deg) + int(lat_min) / 60 + float(lat_sec) / 3600
            lon = int(lon_deg) + int(lon_min) / 60 + float(lon_sec) / 3600
            if lat_dir.upper() == "S":
                lat = -lat
            if lon_dir.upper() == "W":
                lon = -lon
            return lat, lon
        except (ValueError, TypeError):
            return None
    return None

def decimal_to_dms(lat, lon):
    """Convierte decimal a DMS."""
    try:
        lat = float(lat)
        lon = float(lon)
        
        lat_dir = "N" if lat >= 0 else "S"
        abs_lat = abs(lat)
        lat_deg = int(abs_lat)
        lat_min = int((abs_lat - lat_deg) * 60)
        lat_sec = (abs_lat - lat_deg - lat_min / 60) * 3600
        
        lon_dir = "E" if lon >= 0 else "W"
        abs_lon = abs(lon)
        lon_deg = int(abs_lon)
        lon_min = int((abs_lon - lon_deg) * 60)
        lon_sec = (abs_lon - lon_deg - lon_min / 60) * 3600
        
        dms_lat = f"{lat_deg:02d}°{lat_min:02d}'{lat_sec:04.1f}\"{lat_dir}"
        dms_lon = f"{lon_deg:02d}°{lon_min:02d}'{lon_sec:04.1f}\"{lon_dir}"
        return f"{dms_lat} {dms_lon}"
    except (ValueError, TypeError):
        return None

# --- 8. Funciones que actualizan la hoja de cálculo para la conversión de coordenadas ---
def update_decimal_from_dms_sonda(sheet):
    """Convierte DMS a decimal y actualiza las columnas correspondientes para Sondas usando batch_update."""
    try:
        dms_values = sheet.col_values(13)  # Columna M
        if len(dms_values) <= 1:
            st.warning("No se encontraron datos en 'Ubicación sonda google maps'.")
            return
        
        start_row = 2
        batch_updates = []

        for i in range(start_row - 1, len(dms_values)):
            dms = dms_values[i] if i < len(dms_values) else ""
            if dms:
                result = dms_to_decimal(dms)
                if result is not None:
                    lat, lon = result
                    batch_updates.append({"range": f"N{i+1}:O{i+1}", "values": [[lat, lon]]})

        if batch_updates:
            sheet.batch_update(batch_updates)

    except Exception as e:
        st.error(f"Error en la conversión de DMS a decimal para sondas: {str(e)}")

def update_dms_from_decimal_sonda(sheet):
    """Convierte decimal a DMS y actualiza la columna M para Sondas usando batch_update."""
    try:
        lat_values = sheet.col_values(14)  # Columna N
        lon_values = sheet.col_values(15)  # Columna O
        if len(lat_values) <= 1 or len(lon_values) <= 1:
            st.warning("No se encontraron datos en 'Latitud sonda' o 'Longitud Sonda'.")
            return
        
        start_row = 2
        batch_updates = []

        for i in range(start_row - 1, min(len(lat_values), len(lon_values))):
            lat_str = lat_values[i] if i < len(lat_values) else ""
            lon_str = lon_values[i] if i < len(lon_values) else ""
            if lat_str and lon_str:
                try:
                    lat = float(str(lat_str).replace(",", "."))
                    lon = float(str(lon_str).replace(",", "."))
                    dms = decimal_to_dms(lat, lon)
                    if dms:
                        batch_updates.append({"range": f"M{i+1}", "values": [[dms]]})
                except (ValueError, TypeError):
                    pass

        if batch_updates:
            sheet.batch_update(batch_updates)

    except Exception as e:
        st.error(f"Error en la conversión de decimal a DMS para sondas: {str(e)}")
        
            # Actualizar en lotes
            sheet.update_cells(dms_cells[i-1:end_idx-1])
            
            # Mostrar progreso
            progress = min(end_idx / num_rows, 1.0)
            st.progress(progress)
        
        st.success("✅ Conversión de decimal a DMS completada.")
    except Exception as e:
        st.error(f"Error en la conversión de decimal a DMS: {str(e)}")

def update_decimal_from_dms_campo(sheet):
    """Convierte DMS a decimal y actualiza las columnas correspondientes para Campo usando batch_update."""
    try:
        dms_values = sheet.col_values(5)  # Columna E
        if len(dms_values) <= 1:
            st.warning("No se encontraron datos en 'Ubicación campo'.")
            return
        
        start_row = 2
        batch_updates = []

        for i in range(start_row - 1, len(dms_values)):
            dms = dms_values[i] if i < len(dms_values) else ""
            if dms:
                result = dms_to_decimal(dms)
                if result is not None:
                    lat, lon = result
                    batch_updates.append({"range": f"F{i+1}:G{i+1}", "values": [[lat, lon]]})

        if batch_updates:
            sheet.batch_update(batch_updates)

    except Exception as e:
        st.error(f"Error en la conversión de DMS a decimal para campo: {str(e)}")

def update_dms_from_decimal_campo(sheet):
    """Convierte decimal a DMS y actualiza la columna E para Campo usando batch_update."""
    try:
        lat_values = sheet.col_values(6)  # Columna F
        lon_values = sheet.col_values(7)  # Columna G
        if len(lat_values) <= 1 or len(lon_values) <= 1:
            st.warning("No se encontraron datos en 'Latitud campo' o 'Longitud Campo'.")
            return
        
        start_row = 2
        batch_updates = []

        for i in range(start_row - 1, min(len(lat_values), len(lon_values))):
            lat_str = lat_values[i] if i < len(lat_values) else ""
            lon_str = lon_values[i] if i < len(lon_values) else ""
            if lat_str and lon_str:
                try:
                    lat = float(str(lat_str).replace(",", "."))
                    lon = float(str(lon_str).replace(",", "."))
                    dms = decimal_to_dms(lat, lon)
                    if dms:
                        batch_updates.append({"range": f"E{i+1}", "values": [[dms]]})
                except (ValueError, TypeError):
                    pass

        if batch_updates:
            sheet.batch_update(batch_updates)

    except Exception as e:
        st.error(f"Error en la conversión de decimal a DMS para campo: {str(e)}")

# --- 9. Interfaz de usuario en Streamlit ---
def main():
    st.sidebar.title("🧭 Seleccionar Herramienta")
    app_selection = st.sidebar.radio(
        "Elige una herramienta:",
        ["🌍 Georreferenciación de Campos", "📍 Conversión de Coordenadas"]
    )
    
    # Inicializar la conexión a Google Sheets
    client = init_connection()
    if not client:
        st.error("No se pudo establecer conexión con Google Sheets.")
        return
    
    sheet = load_sheet(client)
    if not sheet:
        st.error("No se pudo cargar la hoja de cálculo.")
        return
    
    if app_selection == "🌍 Georreferenciación de Campos":
        display_georreferenciacion(client, sheet)
    else:
        display_conversion_coordenadas(sheet)

def display_georreferenciacion(client, sheet):
    st.title("🌍 Georreferenciación de Campos")
    st.write("""Esta aplicación georreferencia automáticamente las ubicaciones usando archivos shape de Chile y Perú. 
    Si la coordenada está a 100 km de cualquier punto, se catalogará como "OTROS".""")
    
    # Mostrar estado de la descarga de shapefiles
    with st.spinner("Descargando archivos shape..."):
        download_folder = download_shapefiles()
        shapefile_path = os.path.join(download_folder, "combined.shp")
        gdf = load_shapefile(shapefile_path)
        st.success("✅ Archivos shape cargados correctamente")
    
    # Iniciar el proceso de georreferenciación
    if st.button("🚀 Iniciar Proceso de Georreferenciación", type="primary"):
        try:
            with st.spinner("Cargando datos desde Google Sheets..."):
                data = pd.DataFrame(sheet.get_all_records())
                st.success(f"✅ Datos cargados: {len(data)} filas")
            
            # Procesar datos
            with st.spinner("Procesando datos..."):
                final_results, valid_count, invalid_count = process_data(data, gdf)
                
                # Mostrar estadísticas
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total de Registros", len(data))
                with col2:
                    st.metric("Coordenadas Válidas", valid_count)
                with col3:
                    st.metric("Casillas Inválidas", invalid_count)
            
            # Actualizar Google Sheets
            with st.spinner("Actualizando Google Sheets..."):
                if update_google_sheets_georreferencia(client, final_results):
                    st.success("✅ Proceso completado exitosamente!")
                    
                    # Mostrar preview de resultados
                    st.subheader("📊 Previsualización de resultados")
                    st.dataframe(
                        final_results[['original_index', 'Region', 'Provincia', 'Comuna']]
                        .sort_values('original_index')
                        .head(10)
                    )
        
        except Exception as e:
            st.error(f"Error en el proceso: {str(e)}")

def display_conversion_coordenadas(sheet):
    st.title("📍 Conversión de Coordenadas")
    
    tab1, tab2 = st.tabs(["Sondas", "Campo"])
    
    with tab1:
        st.header("Conversión para Sondas")
        st.write("Selecciona la conversión que deseas realizar:")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Convertir DMS a Decimal (Sonda)", 
                         help="Convierte las coordenadas DMS a formato decimal", 
                         key="dms_to_decimal_sonda", 
                         use_container_width=True, 
                         type="primary"):
                update_decimal_from_dms_sonda(sheet)
        with col2:
            if st.button("Convertir Decimal a DMS (Sonda)", 
                         help="Convierte las coordenadas decimales a formato DMS", 
                         key="decimal_to_dms_sonda", 
                         use_container_width=True, 
                         type="primary"):
                update_dms_from_decimal_sonda(sheet)
    
    with tab2:
        st.header("Conversión para Campo")
        st.write("Selecciona la conversión que deseas realizar:")
        
        col3, col4 = st.columns(2)
        with col3:
            if st.button("Convertir DMS a Decimal (Campo)", 
                         help="Convierte las coordenadas DMS a formato decimal para Ubicación campo", 
                         key="dms_to_decimal_campo", 
                         use_container_width=True, 
                         type="primary"):
                update_decimal_from_dms_campo(sheet)
        with col4:
            if st.button("Convertir Decimal a DMS (Campo)", 
                         help="Convierte las coordenadas decimales a formato DMS para Ubicación campo", 
                         key="decimal_to_dms_campo", 
                         use_container_width=True, 
                         type="primary"):
                update_dms_from_decimal_campo(sheet)

if __name__ == "__main__":
    main()
