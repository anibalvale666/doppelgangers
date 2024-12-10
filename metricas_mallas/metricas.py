import open3d as o3d
import numpy as np
import argparse
import logging


# Configurar el logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def compute_triangle_areas(mesh):
    """
    Calcula el área de cada triángulo de la malla.
    """

    triangles = np.asarray(mesh.triangles)  # Índices de los triángulos
    vertices = np.asarray(mesh.vertices)   # Coordenadas de los vértices

    logger.info(f"num triangulos: {len(triangles)}")
    logger.info(f"num Vertices: {len(vertices)}")
    areas = []
    for tri in triangles:
        v0, v1, v2 = vertices[tri]
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        areas.append(area)


    return np.array(areas)


def compute_smoothness(mesh):
    """
    Calcula la suavidad promedio de las normales.
    """
    triangles = np.asarray(mesh.triangles)
    normals = np.asarray(mesh.triangle_normals)


    # Crear un diccionario de vecinos para cada vértice
    vertex_neighbors = {i: set() for i in range(len(mesh.vertices))}
    for tri in triangles:
        vertex_neighbors[tri[0]].update([tri[1], tri[2]])
        vertex_neighbors[tri[1]].update([tri[0], tri[2]])
        vertex_neighbors[tri[2]].update([tri[0], tri[1]])


    # Calcular la suavidad basada en la diferencia de normales entre vecinos
    smoothness = 0.0
    count = 0
    for i, neighbors in vertex_neighbors.items():
        for neighbor in neighbors:
            normal_diff = np.linalg.norm(mesh.vertex_normals[i] - mesh.vertex_normals[neighbor])
            smoothness += normal_diff
            count += 1


    return smoothness / count if count > 0 else 0


def analyze_mesh(file_path):
    logger.info("Cargando la malla desde el archivo...")
    mesh = o3d.io.read_triangle_mesh(file_path)


    if not mesh.has_triangle_normals():
        logger.info("Calculando normales de triángulos...")
        mesh.compute_triangle_normals()


    if not mesh.has_vertex_normals():
        logger.info("Calculando normales de vértices...")
        mesh.compute_vertex_normals()


    logger.info(f"Analizando la malla: {file_path}")


    # Verificar si la malla es válida
    if not mesh.is_edge_manifold() or not mesh.is_vertex_manifold():
        logger.warning("La malla no es completamente válida (puede tener agujeros o bordes no manifold).")


    # Métrica 1: Densidad de vértices
    logger.info("Calculando densidad de vértices...")
    bounding_box = mesh.get_axis_aligned_bounding_box()
    volume = bounding_box.volume()
    num_vertices = np.asarray(mesh.vertices).shape[0]
    vertex_density = num_vertices / volume if volume > 0 else 0
    logger.info(f"Densidad de vértices: {vertex_density:.4f} vértices/unidad cúbica.")


    # Métrica 2: Distribución de áreas de triángulos
    logger.info("Calculando distribución de áreas de los triángulos...")
    triangle_areas = compute_triangle_areas(mesh)
    logger.info(f"Área promedio de los triángulos: {np.mean(triangle_areas):.6f}")
    logger.info(f"Área mínima: {np.min(triangle_areas):.6f}")
    logger.info(f"Área máxima: {np.max(triangle_areas):.6f}")


    # Métrica 3: Suavidad promedio (variación de normales)
    logger.info("Calculando suavidad promedio basada en normales...")
    smoothness = compute_smoothness(mesh)
    logger.info(f"Suavidad promedio (variación de normales): {smoothness:.6f}")


    # Visualización
    logger.info("Visualización: Cierra la ventana para continuar.")
    o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analizar una malla 3D desde un archivo .ply")
    parser.add_argument("file", help="Ruta al archivo .ply que contiene la malla")
    args = parser.parse_args()


    analyze_mesh(args.file)





