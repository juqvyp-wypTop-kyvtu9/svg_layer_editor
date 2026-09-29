import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

if sys.platform != "darwin" and not os.environ.get("DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("QT_OPENGL", "software")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import numpy as np
import pyqtgraph.opengl as gl

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QVector3D
from PySide6.QtWidgets import (
    QApplication,
    QColorDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtSvg import QSvgRenderer

QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseSoftwareOpenGL, True)

TEXTURE_SIZE = 1024
WORLD_SCALE = 0.05


@dataclass
class MapLayer:
    path: str
    image_data: np.ndarray
    item: gl.GLImageItem
    height: float = 0.0
    opacity: int = 100
    name: str = ""
    kind: str = "svg"
    scale_x: float = 1.0
    scale_y: float = 1.0
    rotation: float = 0.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    pivot_x: float = 0.0
    pivot_y: float = 0.0
    billboard: bool = False


def normalize_rgba_image(image: QImage) -> np.ndarray:
    pixels = np.frombuffer(image.constBits(), dtype=np.uint8)
    pixels = pixels.reshape(image.height(), image.width(), 4).copy()
    return np.flipud(pixels).transpose(1, 0, 2).copy()


def svg_to_rgba(path: str) -> np.ndarray:
    renderer = QSvgRenderer(path)
    if not renderer.isValid():
        raise ValueError("有効なSVGファイルではありません。")

    size = renderer.defaultSize()
    if size.width() <= 0 or size.height() <= 0:
        view_box = renderer.viewBoxF()
        if view_box.isNull():
            raise ValueError("SVGの描画サイズを取得できません。")
        size = view_box.size()

    if size.width() <= 0 or size.height() <= 0:
        raise ValueError("SVGの表示サイズを取得できません。")

    aspect = size.width() / size.height()
    target_width = TEXTURE_SIZE
    target_height = TEXTURE_SIZE
    if aspect >= 1.0:
        target_height = max(1, int(TEXTURE_SIZE / aspect))
    else:
        target_width = max(1, int(TEXTURE_SIZE * aspect))

    x = (TEXTURE_SIZE - target_width) / 2
    y = (TEXTURE_SIZE - target_height) / 2
    target = QRectF(x, y, target_width, target_height)

    image = QImage(TEXTURE_SIZE, TEXTURE_SIZE, QImage.Format.Format_RGBA8888)
    image.fill(Qt.GlobalColor.transparent)

    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
    try:
        renderer.render(painter, target)
    finally:
        painter.end()

    return normalize_rgba_image(image)


def text_to_rgba(text: str, font_size: int = 72, color: tuple[int, int, int, int] = (255, 255, 255, 255)) -> np.ndarray:
    text = text or "TEXT"
    font = QFont("Arial", font_size, QFont.Weight.Bold)
    image = QImage(TEXTURE_SIZE, TEXTURE_SIZE, QImage.Format.Format_RGBA8888)
    image.fill(Qt.GlobalColor.transparent)

    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    painter.setPen(QColor(*color))
    painter.setFont(font)

    metrics = painter.fontMetrics()
    rect = metrics.boundingRect(text)
    x = (TEXTURE_SIZE - rect.width()) / 2
    y = (TEXTURE_SIZE + rect.height()) / 2
    painter.drawText(int(x), int(y), text)
    painter.end()

    return normalize_rgba_image(image)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SVG階層マップエディタ")
        self.resize(1500, 900)

        self.layers: list[MapLayer] = []
        self.background_item = None
        self.background_path = ""
        self.dragging = False
        self.drag_layer = None
        self._drag_start_screen = None
        self._drag_start_world = None

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor(QColor("#ffffff"))
        self.view.setCameraPosition(distance=150, elevation=35, azimuth=45)
        self.view.mousePressEvent = self._mouse_press_event
        self.view.mouseMoveEvent = self._mouse_move_event
        self.view.mouseReleaseEvent = self._mouse_release_event

        self.show_axis = True
        self.axis_item = gl.GLAxisItem()
        self.axis_item.setSize(x=200, y=200, z=200)
        self.view.addItem(self.axis_item)

        self.layer_list = QListWidget()
        self.layer_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.layer_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.layer_list.model().rowsMoved.connect(self.handle_rows_moved)
        self.layer_list.currentRowChanged.connect(self.select_layer)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("レイヤー名")
        self.name_edit.editingFinished.connect(self.rename_selected_layer)

        self.add_button = QPushButton("SVGを追加")
        self.add_button.clicked.connect(self.add_svg)

        self.add_icon_button = QPushButton("SVGアイコン追加")
        self.add_icon_button.clicked.connect(self.add_icon_svg)

        self.add_text_button = QPushButton("テキスト追加")
        self.add_text_button.clicked.connect(self.add_text)

        self.background_button = QPushButton("背景 PNG を読込")
        self.background_button.clicked.connect(self.load_background_png)

        self.background_color_button = QPushButton("背景色を選択")
        self.background_color_button.clicked.connect(self.choose_background_color)

        self.axis_toggle_button = QPushButton("軸表示: ON")
        self.axis_toggle_button.clicked.connect(self.toggle_axis_visibility)

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("テキストを入力")
        self.text_input.setText("Layer")

        self.load_button = QPushButton("JSON読込")
        self.load_button.clicked.connect(self.load_project)

        self.save_button = QPushButton("JSON保存")
        self.save_button.clicked.connect(self.save_project)

        self.export_button = QPushButton("PNG出力")
        self.export_button.clicked.connect(self.export_png)

        self.move_up_button = QPushButton("↑ 上へ")
        self.move_up_button.clicked.connect(lambda: self.move_selected_layer(-1))

        self.move_down_button = QPushButton("↓ 下へ")
        self.move_down_button.clicked.connect(lambda: self.move_selected_layer(1))

        self.remove_button = QPushButton("削除")
        self.remove_button.clicked.connect(self.remove_selected_layer)

        self.offset_x_spin = QDoubleSpinBox()
        self.offset_x_spin.setRange(-1000.0, 1000.0)
        self.offset_x_spin.setSingleStep(1.0)
        self.offset_x_spin.valueChanged.connect(self.change_offset)

        self.offset_y_spin = QDoubleSpinBox()
        self.offset_y_spin.setRange(-1000.0, 1000.0)
        self.offset_y_spin.setSingleStep(1.0)
        self.offset_y_spin.valueChanged.connect(self.change_offset)

        self.pivot_x_spin = QDoubleSpinBox()
        self.pivot_x_spin.setRange(-1000.0, 1000.0)
        self.pivot_x_spin.setSingleStep(1.0)
        self.pivot_x_spin.valueChanged.connect(self.change_pivot)

        self.pivot_y_spin = QDoubleSpinBox()
        self.pivot_y_spin.setRange(-1000.0, 1000.0)
        self.pivot_y_spin.setSingleStep(1.0)
        self.pivot_y_spin.valueChanged.connect(self.change_pivot)

        self.scale_x_spin = QDoubleSpinBox()
        self.scale_x_spin.setRange(0.01, 20.0)
        self.scale_x_spin.setSingleStep(0.1)
        self.scale_x_spin.setValue(1.0)
        self.scale_x_spin.valueChanged.connect(self.change_scale)

        self.scale_y_spin = QDoubleSpinBox()
        self.scale_y_spin.setRange(0.01, 20.0)
        self.scale_y_spin.setSingleStep(0.1)
        self.scale_y_spin.setValue(1.0)
        self.scale_y_spin.valueChanged.connect(self.change_scale)

        self.rotation_spin = QDoubleSpinBox()
        self.rotation_spin.setRange(-360.0, 360.0)
        self.rotation_spin.setSingleStep(5.0)
        self.rotation_spin.setValue(0.0)
        self.rotation_spin.valueChanged.connect(self.change_rotation)

        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(-100.0, 100.0)
        self.height_spin.setSingleStep(1.0)
        self.height_spin.setSuffix(" 3D単位")
        self.height_spin.valueChanged.connect(self.change_height)

        self.height_slider = QSlider(Qt.Orientation.Horizontal)
        self.height_slider.setRange(-1000, 1000)
        self.height_slider.setSingleStep(1)
        self.height_slider.valueChanged.connect(self.change_height_from_slider)

        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.valueChanged.connect(self.change_opacity)

        self.opacity_label = QLabel("100%")

        self.scale_x_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_x_slider.setRange(1, 200)
        self.scale_x_slider.setValue(100)
        self.scale_x_slider.valueChanged.connect(self.change_scale_from_slider)

        self.scale_y_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_y_slider.setRange(1, 200)
        self.scale_y_slider.setValue(100)
        self.scale_y_slider.valueChanged.connect(self.change_scale_from_slider)

        self.rotation_slider = QSlider(Qt.Orientation.Horizontal)
        self.rotation_slider.setRange(-180, 180)
        self.rotation_slider.setValue(0)
        self.rotation_slider.valueChanged.connect(self.change_rotation_from_slider)

        self.azimuth_spin = QDoubleSpinBox()
        self.azimuth_spin.setRange(0.0, 360.0)
        self.azimuth_spin.setSingleStep(5.0)
        self.azimuth_spin.setSuffix("°")
        self.azimuth_spin.valueChanged.connect(self.change_camera_from_controls)

        self.azimuth_slider = QSlider(Qt.Orientation.Horizontal)
        self.azimuth_slider.setRange(0, 360)
        self.azimuth_slider.valueChanged.connect(self.change_camera_slider)

        self.elevation_spin = QDoubleSpinBox()
        self.elevation_spin.setRange(-90.0, 90.0)
        self.elevation_spin.setSingleStep(5.0)
        self.elevation_spin.setSuffix("°")
        self.elevation_spin.valueChanged.connect(self.change_camera_from_controls)

        self.elevation_slider = QSlider(Qt.Orientation.Horizontal)
        self.elevation_slider.setRange(-90, 90)
        self.elevation_slider.valueChanged.connect(self.change_camera_slider)

        self.distance_spin = QDoubleSpinBox()
        self.distance_spin.setRange(10.0, 1000.0)
        self.distance_spin.setSingleStep(10.0)
        self.distance_spin.setSuffix(" units")
        self.distance_spin.valueChanged.connect(self.change_camera_from_controls)

        self.distance_slider = QSlider(Qt.Orientation.Horizontal)
        self.distance_slider.setRange(10, 1000)
        self.distance_slider.valueChanged.connect(self.change_camera_slider)

        self.reset_view_button = QPushButton("視点リセット")
        self.reset_view_button.clicked.connect(self.reset_camera_view)

        self.iso_button = QPushButton("アイソ")
        self.iso_button.clicked.connect(lambda: self.set_camera_preset("iso"))

        self.front_button = QPushButton("正面")
        self.front_button.clicked.connect(lambda: self.set_camera_preset("front"))

        self.top_button = QPushButton("上面")
        self.top_button.clicked.connect(lambda: self.set_camera_preset("top"))

        toolbar = QGridLayout()
        toolbar.setColumnStretch(0, 1)
        toolbar.setColumnStretch(1, 1)
        toolbar.setColumnStretch(2, 1)
        toolbar.setColumnStretch(3, 1)
        toolbar.setColumnStretch(4, 1)
        toolbar.setColumnStretch(5, 1)
        toolbar.setColumnStretch(6, 1)

        toolbar_buttons = [
            self.add_button,
            self.add_icon_button,
            self.add_text_button,
            self.background_button,
            self.background_color_button,
            self.axis_toggle_button,
            self.load_button,
            self.save_button,
            self.export_button,
        ]
        for index, button in enumerate(toolbar_buttons):
            row = index // 2
            col = index % 2
            toolbar.addWidget(button, row, col)

        layer_action_layout = QHBoxLayout()
        layer_action_layout.addWidget(self.move_up_button)
        layer_action_layout.addWidget(self.move_down_button)
        layer_action_layout.addWidget(self.remove_button)

        camera_layout = QHBoxLayout()
        camera_layout.addWidget(self.iso_button)
        camera_layout.addWidget(self.front_button)
        camera_layout.addWidget(self.top_button)
        camera_layout.addWidget(self.reset_view_button)

        side_layout = QVBoxLayout()
        side_layout.addLayout(toolbar)
        side_layout.addWidget(QLabel("レイヤー"))
        side_layout.addWidget(self.layer_list)
        side_layout.addLayout(layer_action_layout)
        side_layout.addWidget(QLabel("レイヤー名"))
        side_layout.addWidget(self.name_edit)
        side_layout.addWidget(QLabel("テキスト"))
        side_layout.addWidget(self.text_input)
        side_layout.addWidget(QLabel("位置 X"))
        side_layout.addWidget(self.offset_x_spin)
        side_layout.addWidget(QLabel("位置 Y"))
        side_layout.addWidget(self.offset_y_spin)
        side_layout.addWidget(QLabel("回転中心 X"))
        side_layout.addWidget(self.pivot_x_spin)
        side_layout.addWidget(QLabel("回転中心 Y"))
        side_layout.addWidget(self.pivot_y_spin)
        side_layout.addWidget(QLabel("拡大縮小 X"))
        side_layout.addWidget(self.scale_x_spin)
        side_layout.addWidget(self.scale_x_slider)
        side_layout.addWidget(QLabel("拡大縮小 Y"))
        side_layout.addWidget(self.scale_y_spin)
        side_layout.addWidget(self.scale_y_slider)
        side_layout.addWidget(QLabel("回転"))
        side_layout.addWidget(self.rotation_spin)
        side_layout.addWidget(self.rotation_slider)
        side_layout.addWidget(QLabel("高さ"))
        side_layout.addWidget(self.height_spin)
        side_layout.addWidget(self.height_slider)
        side_layout.addWidget(QLabel("透明度（不透明度）"))
        side_layout.addWidget(self.opacity_slider)
        side_layout.addWidget(self.opacity_label)
        side_layout.addWidget(QLabel("3Dカメラ"))
        side_layout.addLayout(camera_layout)
        side_layout.addWidget(QLabel("方位角"))
        side_layout.addWidget(self.azimuth_spin)
        side_layout.addWidget(self.azimuth_slider)
        side_layout.addWidget(QLabel("仰角"))
        side_layout.addWidget(self.elevation_spin)
        side_layout.addWidget(self.elevation_slider)
        side_layout.addWidget(QLabel("距離"))
        side_layout.addWidget(self.distance_spin)
        side_layout.addWidget(self.distance_slider)

        side_panel = QWidget()
        side_panel.setLayout(side_layout)
        side_panel.setFixedWidth(360)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(side_panel)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        root_layout = QHBoxLayout()
        root_layout.addWidget(scroll)
        root_layout.addWidget(self.view, 1)

        root = QWidget()
        root.setLayout(root_layout)
        self.setCentralWidget(root)

        self.disable_layer_controls()
        self.update_camera_controls()

    def disable_layer_controls(self):
        self.name_edit.setEnabled(False)
        self.offset_x_spin.setEnabled(False)
        self.offset_y_spin.setEnabled(False)
        self.pivot_x_spin.setEnabled(False)
        self.pivot_y_spin.setEnabled(False)
        self.scale_x_spin.setEnabled(False)
        self.scale_y_spin.setEnabled(False)
        self.scale_x_slider.setEnabled(False)
        self.scale_y_slider.setEnabled(False)
        self.rotation_spin.setEnabled(False)
        self.rotation_slider.setEnabled(False)
        self.height_spin.setEnabled(False)
        self.height_slider.setEnabled(False)
        self.opacity_slider.setEnabled(False)
        self.move_up_button.setEnabled(False)
        self.move_down_button.setEnabled(False)
        self.remove_button.setEnabled(False)

    def selected_layer(self):
        item = self.layer_list.currentItem()
        if item is None:
            return None
        layer = item.data(Qt.ItemDataRole.UserRole)
        return layer if isinstance(layer, MapLayer) else None

    def add_layer_to_list(self, layer: MapLayer):
        list_item = QListWidgetItem(layer.name)
        list_item.setData(Qt.ItemDataRole.UserRole, layer)
        self.layer_list.addItem(list_item)
        self.layer_list.setCurrentItem(list_item)

    def apply_layer_transform(self, layer: MapLayer):
        half_size = TEXTURE_SIZE * WORLD_SCALE / 2
        layer.item.resetTransform()
        layer.item.translate(layer.offset_x, layer.offset_y, layer.height)

        if layer.billboard:
            camera_azimuth = self.view.cameraParams()["azimuth"]
            layer.item.rotate(-(camera_azimuth + layer.rotation), 0, 0, 1)
        else:
            layer.item.rotate(layer.rotation, 0, 0, 1)

        layer.item.translate(layer.pivot_x, layer.pivot_y, 0)
        layer.item.scale(layer.scale_x * WORLD_SCALE, layer.scale_y * WORLD_SCALE, 1)
        layer.item.translate(-half_size, -half_size, 0)

    def update_layer_controls(self):
        layer = self.selected_layer()
        if layer is None:
            self.disable_layer_controls()
            return

        self.name_edit.setEnabled(True)
        self.offset_x_spin.setEnabled(True)
        self.offset_y_spin.setEnabled(True)
        self.pivot_x_spin.setEnabled(True)
        self.pivot_y_spin.setEnabled(True)
        self.scale_x_spin.setEnabled(True)
        self.scale_y_spin.setEnabled(True)
        self.scale_x_slider.setEnabled(True)
        self.scale_y_slider.setEnabled(True)
        self.rotation_spin.setEnabled(True)
        self.rotation_slider.setEnabled(True)
        self.height_spin.setEnabled(True)
        self.height_slider.setEnabled(True)
        self.opacity_slider.setEnabled(True)

        row = self.layer_list.currentRow()
        self.move_up_button.setEnabled(row > 0)
        self.move_down_button.setEnabled(row < self.layer_list.count() - 1)
        self.remove_button.setEnabled(True)

        self.name_edit.blockSignals(True)
        self.offset_x_spin.blockSignals(True)
        self.offset_y_spin.blockSignals(True)
        self.pivot_x_spin.blockSignals(True)
        self.pivot_y_spin.blockSignals(True)
        self.scale_x_spin.blockSignals(True)
        self.scale_y_spin.blockSignals(True)
        self.scale_x_slider.blockSignals(True)
        self.scale_y_slider.blockSignals(True)
        self.rotation_spin.blockSignals(True)
        self.rotation_slider.blockSignals(True)
        self.height_spin.blockSignals(True)
        self.height_slider.blockSignals(True)
        self.opacity_slider.blockSignals(True)

        self.name_edit.setText(layer.name)
        self.offset_x_spin.setValue(layer.offset_x)
        self.offset_y_spin.setValue(layer.offset_y)
        self.pivot_x_spin.setValue(layer.pivot_x)
        self.pivot_y_spin.setValue(layer.pivot_y)
        self.scale_x_spin.setValue(layer.scale_x)
        self.scale_y_spin.setValue(layer.scale_y)
        self.scale_x_slider.setValue(int(layer.scale_x * 100))
        self.scale_y_slider.setValue(int(layer.scale_y * 100))
        self.rotation_spin.setValue(layer.rotation)
        self.rotation_slider.setValue(int(layer.rotation))
        self.height_spin.setValue(layer.height)
        self.height_slider.setValue(int(layer.height * 10))
        self.opacity_slider.setValue(layer.opacity)
        self.opacity_label.setText(f"{layer.opacity}%")

        self.name_edit.blockSignals(False)
        self.offset_x_spin.blockSignals(False)
        self.offset_y_spin.blockSignals(False)
        self.pivot_x_spin.blockSignals(False)
        self.pivot_y_spin.blockSignals(False)
        self.scale_x_spin.blockSignals(False)
        self.scale_y_spin.blockSignals(False)
        self.scale_x_slider.blockSignals(False)
        self.scale_y_slider.blockSignals(False)
        self.rotation_spin.blockSignals(False)
        self.rotation_slider.blockSignals(False)
        self.height_spin.blockSignals(False)
        self.height_slider.blockSignals(False)
        self.opacity_slider.blockSignals(False)

    def select_layer(self, row):
        self.update_layer_controls()

    def rename_selected_layer(self):
        layer = self.selected_layer()
        if layer is None:
            return
        new_name = self.name_edit.text().strip() or layer.name
        layer.name = new_name
        current_item = self.layer_list.currentItem()
        if current_item is not None:
            current_item.setText(new_name)

    def add_svg_layer(self, path: str, *, name: str | None = None, scale_x: float = 1.0, scale_y: float = 1.0, rotation: float = 0.0, height: float | None = None, opacity: int = 100, offset_x: float = 0.0, offset_y: float = 0.0, pivot_x: float = 0.0, pivot_y: float = 0.0, billboard: bool = False):
        original_data = svg_to_rgba(path)
        item = gl.GLImageItem(data=original_data, smooth=True, glOptions="translucent")
        base_name = name or Path(path).name
        layer = MapLayer(
            path=path,
            image_data=original_data,
            item=item,
            height=len(self.layers) * 5.0 if height is None else height,
            opacity=opacity,
            name=base_name,
            kind="svg",
            scale_x=scale_x,
            scale_y=scale_y,
            rotation=rotation,
            offset_x=offset_x,
            offset_y=offset_y,
            pivot_x=pivot_x,
            pivot_y=pivot_y,
            billboard=billboard,
        )
        self.layers.append(layer)
        self.view.addItem(item)
        self.apply_layer_transform(layer)
        self.add_layer_to_list(layer)

    def add_icon_svg(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "SVGアイコンを選択", "", "SVGファイル (*.svg)")
        for path in paths:
            try:
                self.add_svg_layer(path, name=Path(path).stem, scale_x=0.7, scale_y=0.7, height=len(self.layers) * 4.0, billboard=True)
            except Exception as exc:
                QMessageBox.warning(self, "読込エラー", f"{path}\n\n{exc}")

    def add_svg(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "SVGを選択", "", "SVGファイル (*.svg)")
        for path in paths:
            try:
                self.add_svg_layer(path)
            except Exception as exc:
                QMessageBox.warning(self, "読込エラー", f"{path}\n\n{exc}")

    def add_text(self):
        text = self.text_input.text().strip() or "TEXT"
        image_data = text_to_rgba(text, font_size=72)
        item = gl.GLImageItem(data=image_data, smooth=True, glOptions="translucent")
        layer = MapLayer(
            path="",
            image_data=image_data,
            item=item,
            height=len(self.layers) * 5.0,
            opacity=100,
            name=text,
            kind="text",
            scale_x=1.0,
            scale_y=1.0,
            rotation=0.0,
            offset_x=0.0,
            offset_y=0.0,
            billboard=True,
        )
        self.layers.append(layer)
        self.view.addItem(item)
        self.apply_layer_transform(layer)
        self.add_layer_to_list(layer)

    def load_background_png(self):
        path, _ = QFileDialog.getOpenFileName(self, "背景PNGを選択", "", "PNG画像 (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return

        try:
            image = QImage(path)
            if image.isNull():
                raise ValueError("画像を読み込めませんでした。")
            if self.background_item is not None:
                self.view.removeItem(self.background_item)
            rgba = normalize_rgba_image(image)
            self.background_item = gl.GLImageItem(data=rgba, smooth=True, glOptions="translucent")
            self.background_item.setData(rgba)
            self.background_item.scale(0.2, 0.2, 1)
            self.background_item.translate(-TEXTURE_SIZE * WORLD_SCALE / 2, -TEXTURE_SIZE * WORLD_SCALE / 2, -50)
            self.view.addItem(self.background_item)
            self.background_path = path
        except Exception as exc:
            QMessageBox.warning(self, "背景読み込みエラー", str(exc))

    def get_background_color(self):
        bgcolor = self.view.opts.get("bgcolor")
        if bgcolor is None:
            return QColor("white")
        try:
            return QColor.fromRgbF(*bgcolor)
        except Exception:
            return QColor("white")

    def choose_background_color(self):
        color = QColorDialog.getColor(self.get_background_color(), self, "背景色を選択")
        if not color.isValid():
            return
        self.view.setBackgroundColor(color)
        self.view.update()

    def toggle_axis_visibility(self):
        self.show_axis = not self.show_axis
        self.axis_item.setVisible(self.show_axis)
        if hasattr(self.axis_item, "lineplot") and self.axis_item.lineplot is not None:
            self.axis_item.lineplot.setVisible(self.show_axis)

        if self.show_axis:
            if self.axis_item not in list(self.view.items):
                self.view.addItem(self.axis_item)
            self.axis_toggle_button.setText("軸表示: ON")
        else:
            if self.axis_item in list(self.view.items):
                self.view.removeItem(self.axis_item)
            self.axis_toggle_button.setText("軸表示: OFF")
        self.view.update()

    def refresh_view_order(self):
        for item in list(self.view.items):
            if isinstance(item, gl.GLImageItem):
                self.view.removeItem(item)
        if self.background_item is not None:
            self.view.addItem(self.background_item)
        for layer in self.layers:
            self.view.addItem(layer.item)
        if self.show_axis:
            if self.axis_item not in list(self.view.items):
                self.view.addItem(self.axis_item)

    def handle_rows_moved(self, parent, start, end, destination, row):
        reordered = []
        for i in range(self.layer_list.count()):
            item = self.layer_list.item(i)
            if item is not None:
                layer = item.data(Qt.ItemDataRole.UserRole)
                if isinstance(layer, MapLayer):
                    reordered.append(layer)
        self.layers = reordered
        self.refresh_view_order()

    def move_selected_layer(self, delta: int):
        row = self.layer_list.currentRow()
        if not (0 <= row < self.layer_list.count()):
            return
        target_row = row + delta
        if not (0 <= target_row < self.layer_list.count()):
            return

        item = self.layer_list.takeItem(row)
        self.layer_list.insertItem(target_row, item)
        self.layer_list.setCurrentRow(target_row)

        reordered = []
        for i in range(self.layer_list.count()):
            li = self.layer_list.item(i)
            if li is not None:
                layer = li.data(Qt.ItemDataRole.UserRole)
                if isinstance(layer, MapLayer):
                    reordered.append(layer)
        self.layers = reordered
        self.refresh_view_order()

    def remove_selected_layer(self):
        row = self.layer_list.currentRow()
        if not (0 <= row < self.layer_list.count()):
            return
        item = self.layer_list.takeItem(row)
        layer = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        if isinstance(layer, MapLayer):
            self.view.removeItem(layer.item)
            self.layers.remove(layer)
        if self.layers:
            next_row = min(row, len(self.layers) - 1)
            self.layer_list.setCurrentRow(next_row)
        else:
            self.disable_layer_controls()

    def update_item_position(self, layer: MapLayer):
        self.apply_layer_transform(layer)
        self.view.update()

    def change_height(self, value):
        layer = self.selected_layer()
        if layer is None:
            return
        layer.height = value
        self.update_item_position(layer)

    def change_opacity(self, value):
        layer = self.selected_layer()
        if layer is None:
            return
        layer.opacity = value
        self.opacity_label.setText(f"{value}%")
        data = layer.image_data.copy()
        data[:, :, 3] = (data[:, :, 3].astype(np.uint16) * value // 100).astype(np.uint8)
        layer.item.setData(data)

    def change_offset(self, value):
        layer = self.selected_layer()
        if layer is None:
            return
        if self.sender() == self.offset_x_spin:
            layer.offset_x = value
        elif self.sender() == self.offset_y_spin:
            layer.offset_y = value
        self.apply_layer_transform(layer)
        self.view.update()

    def change_pivot(self, value):
        layer = self.selected_layer()
        if layer is None:
            return
        if self.sender() == self.pivot_x_spin:
            layer.pivot_x = value
        elif self.sender() == self.pivot_y_spin:
            layer.pivot_y = value
        self.apply_layer_transform(layer)
        self.view.update()

    def change_scale(self, value):
        layer = self.selected_layer()
        if layer is None:
            return
        if self.sender() == self.scale_x_spin:
            layer.scale_x = value
        elif self.sender() == self.scale_y_spin:
            layer.scale_y = value
        self.apply_layer_transform(layer)
        self.view.update()

    def change_rotation(self, value):
        layer = self.selected_layer()
        if layer is None:
            return
        layer.rotation = value
        self.apply_layer_transform(layer)
        self.view.update()

    def change_height_from_slider(self, value):
        layer = self.selected_layer()
        if layer is None:
            return
        layer.height = value / 10.0
        self.height_spin.setValue(layer.height)
        self.apply_layer_transform(layer)
        self.view.update()

    def change_scale_from_slider(self, value):
        layer = self.selected_layer()
        if layer is None:
            return
        scale = value / 100.0
        if self.sender() == self.scale_x_slider:
            layer.scale_x = scale
            self.scale_x_spin.setValue(scale)
        else:
            layer.scale_y = scale
            self.scale_y_spin.setValue(scale)
        self.apply_layer_transform(layer)
        self.view.update()

    def change_rotation_from_slider(self, value):
        layer = self.selected_layer()
        if layer is None:
            return
        layer.rotation = float(value)
        self.rotation_spin.setValue(layer.rotation)
        self.apply_layer_transform(layer)
        self.view.update()

    def change_camera_slider(self, value):
        if self.sender() == self.azimuth_slider:
            self.azimuth_spin.setValue(float(value))
        elif self.sender() == self.elevation_slider:
            self.elevation_spin.setValue(float(value))
        elif self.sender() == self.distance_slider:
            self.distance_spin.setValue(float(value))
        self.change_camera_from_controls(value)

    def event(self, event):
        if event.type() == event.Type.Gesture:
            if event.gestureType() == Qt.GestureType.PinchGesture:
                pinch = event.gesture(Qt.PinchGesture)
                if pinch is not None:
                    scale_factor = pinch.scaleFactor()
                    if scale_factor != 0:
                        camera = self.view.cameraParams()
                        new_distance = max(10.0, min(1000.0, camera["distance"] / scale_factor))
                        self.view.setCameraPosition(distance=new_distance, elevation=camera["elevation"], azimuth=camera["azimuth"])
                        self.update_camera_controls()
                        return True
        return super().event(event)

    def _pick_layer_at(self, pos):
        if not self.layers:
            return None
        selected = self.selected_layer()
        if selected is not None:
            return selected
        return self.layers[-1]

    def _screen_to_world_xy(self, x, y, z=0.0):
        if self._drag_start_screen is None or self._drag_start_world is None:
            return 0.0, 0.0

        dx = x - self._drag_start_screen.x()
        dy = self._drag_start_screen.y() - y
        camera_distance = self.view.cameraParams().get("distance", 150.0)
        zoom = max(0.2, camera_distance / 200.0)
        world_x = self._drag_start_world[0] + dx * 0.15 * zoom
        world_y = self._drag_start_world[1] + dy * 0.15 * zoom
        return world_x, world_y

    def _mouse_press_event(self, event):
        gl.GLViewWidget.mousePressEvent(self.view, event)
        layer = self._pick_layer_at(event.position())
        if layer is not None:
            self.layer_list.setCurrentRow(self.layers.index(layer))
            self.dragging = True
            self.drag_layer = layer
            self._drag_start_screen = event.position()
            self._drag_start_world = (layer.offset_x, layer.offset_y)

    def _mouse_move_event(self, event):
        gl.GLViewWidget.mouseMoveEvent(self.view, event)
        if not self.dragging or self.drag_layer is None:
            return
        wx, wy = self._screen_to_world_xy(event.position().x(), event.position().y(), self.drag_layer.height)
        self.drag_layer.offset_x = wx
        self.drag_layer.offset_y = wy
        self.offset_x_spin.setValue(self.drag_layer.offset_x)
        self.offset_y_spin.setValue(self.drag_layer.offset_y)
        self.apply_layer_transform(self.drag_layer)
        self.view.update()

    def _mouse_release_event(self, event):
        gl.GLViewWidget.mouseReleaseEvent(self.view, event)
        self.dragging = False
        self.drag_layer = None
        self._drag_start_screen = None
        self._drag_start_world = None

    def reset_camera_view(self):
        self.view.setCameraPosition(distance=150, elevation=35, azimuth=45)
        self.azimuth_spin.setValue(45)
        self.elevation_spin.setValue(35)
        self.distance_spin.setValue(150)

    def set_camera_preset(self, preset: str):
        if preset == "iso":
            self.view.setCameraPosition(distance=150, elevation=35, azimuth=45)
            self.azimuth_spin.setValue(45)
            self.elevation_spin.setValue(35)
            self.distance_spin.setValue(150)
        elif preset == "front":
            self.view.setCameraPosition(distance=220, elevation=0, azimuth=0)
            self.azimuth_spin.setValue(0)
            self.elevation_spin.setValue(0)
            self.distance_spin.setValue(220)
        elif preset == "top":
            self.view.setCameraPosition(distance=220, elevation=90, azimuth=0)
            self.azimuth_spin.setValue(0)
            self.elevation_spin.setValue(90)
            self.distance_spin.setValue(220)

    def update_camera_controls(self):
        self.azimuth_spin.blockSignals(True)
        self.elevation_spin.blockSignals(True)
        self.distance_spin.blockSignals(True)
        camera = self.view.cameraParams()
        self.azimuth_spin.setValue(float(camera["azimuth"]))
        self.elevation_spin.setValue(float(camera["elevation"]))
        self.distance_spin.setValue(float(camera["distance"]))
        self.azimuth_spin.blockSignals(False)
        self.elevation_spin.blockSignals(False)
        self.distance_spin.blockSignals(False)

    def change_camera_from_controls(self, value):
        self.view.setCameraPosition(
            distance=self.distance_spin.value(),
            elevation=self.elevation_spin.value(),
            azimuth=self.azimuth_spin.value(),
        )
        for layer in self.layers:
            self.apply_layer_transform(layer)

    def save_project(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "プロジェクトを保存", "", "JSONファイル (*.json)")
        if not file_path:
            return
        payload = {
            "version": 1,
            "layers": [
                {
                    "kind": layer.kind,
                    "path": layer.path,
                    "name": layer.name,
                    "height": float(layer.height),
                    "opacity": int(layer.opacity),
                    "scale_x": float(layer.scale_x),
                    "scale_y": float(layer.scale_y),
                    "rotation": float(layer.rotation),
                    "offset_x": float(layer.offset_x),
                    "offset_y": float(layer.offset_y),
                    "billboard": bool(layer.billboard),
                    "text": layer.name if layer.kind == "text" else "",
                }
                for layer in self.layers
            ],
        }
        with open(file_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    def load_project(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "JSONプロジェクトを読み込み", "", "JSONファイル (*.json)")
        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as exc:
            QMessageBox.warning(self, "読み込みエラー", f"{exc}")
            return

        self.clear_layers()
        for layer_data in payload.get("layers", []):
            kind = layer_data.get("kind", "svg")
            name = layer_data.get("name", "Layer")
            height = float(layer_data.get("height", 0.0))
            opacity = int(layer_data.get("opacity", 100))
            scale_x = float(layer_data.get("scale_x", 1.0))
            scale_y = float(layer_data.get("scale_y", 1.0))
            rotation = float(layer_data.get("rotation", 0.0))
            offset_x = float(layer_data.get("offset_x", 0.0))
            offset_y = float(layer_data.get("offset_y", 0.0))
            billboard = bool(layer_data.get("billboard", False))

            try:
                if kind == "text":
                    text_value = layer_data.get("text") or name
                    image_data = text_to_rgba(text_value)
                    item = gl.GLImageItem(data=image_data, smooth=True, glOptions="translucent")
                    layer = MapLayer(
                        path="",
                        image_data=image_data,
                        item=item,
                        height=height,
                        opacity=opacity,
                        name=name,
                        kind="text",
                        scale_x=scale_x,
                        scale_y=scale_y,
                        rotation=rotation,
                        offset_x=offset_x,
                        offset_y=offset_y,
                        billboard=billboard,
                    )
                    self.layers.append(layer)
                    self.view.addItem(item)
                    self.apply_layer_transform(layer)
                    self.add_layer_to_list(layer)
                else:
                    path = layer_data.get("path")
                    if not path:
                        continue
                    self.add_svg_layer(path, name=name, scale_x=scale_x, scale_y=scale_y, rotation=rotation, height=height, opacity=opacity, offset_x=offset_x, offset_y=offset_y, billboard=billboard)
            except Exception as exc:
                QMessageBox.warning(self, "読み込みエラー", f"{name}\n\n{exc}")

    def clear_layers(self):
        for layer in list(self.layers):
            self.view.removeItem(layer.item)
        self.layers.clear()
        self.layer_list.clear()
        self.disable_layer_controls()

    def export_png(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "PNGとして出力", "scene.png", "PNG画像 (*.png)")
        if not file_path:
            return
        image = self.view.grabFramebuffer()
        if image is None:
            QMessageBox.warning(self, "出力エラー", "画面のキャプチャ取得に失敗しました。")
            return
        if not image.save(file_path):
            QMessageBox.warning(self, "出力エラー", "PNGファイルの保存に失敗しました。")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())