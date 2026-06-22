from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem,
    QLabel, QLineEdit, QMenu,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QBrush
from ..models.cluster import Cluster


class ClusterPanel(QWidget):
    cluster_selected        = Signal(int)   # cluster_id  (-1 = all)
    cluster_delete_requested = Signal(int)  # cluster_id
    cluster_autolabel_requested = Signal(int)  # cluster_id  (YOLOE visual)
    cluster_sam3_requested  = Signal(int)    # cluster_id  (SAM 3 texto)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._db = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        header = QLabel("Grupos / Clusters")
        header.setStyleSheet("font-weight: bold; font-size: 13px; padding: 4px;")
        layout.addWidget(header)

        self._search = QLineEdit()
        self._search.setPlaceholderText("Filtrar grupos...")
        self._search.textChanged.connect(self._filter)
        layout.addWidget(self._search)

        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._context_menu)
        self._tree.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._tree)

    def set_db(self, db):
        self._db = db
        self._status_filter = "todas"
        self.refresh()

    def set_status_filter(self, status: str):
        self._status_filter = status
        self.refresh()

    def refresh(self):
        if self._db is None:
            return
        status = getattr(self, "_status_filter", "todas")
        self._tree.clear()

        all_item = QTreeWidgetItem(["Todas las imágenes"])
        all_item.setData(0, Qt.UserRole, ("all", -1))
        self._tree.addTopLevelItem(all_item)

        for row in self._db.get_all_clusters(status):
            cluster = Cluster.from_db_row(row)
            if cluster.image_count == 0:
                continue          # hide clusters with no matching images
            label = f"{cluster.name}  ({cluster.image_count})"
            item = QTreeWidgetItem([label])
            item.setData(0, Qt.UserRole, ("cluster", cluster.id))
            color = QColor(cluster.color)
            color.setAlpha(180)
            item.setBackground(0, QBrush(color))
            item.setForeground(0, QBrush(QColor("#FFFFFF")))
            self._tree.addTopLevelItem(item)

        self._filter(self._search.text())

    def _filter(self, text: str):
        text = text.lower()
        root = self._tree.invisibleRootItem()
        for i in range(root.childCount()):
            item = root.child(i)
            item.setHidden(bool(text) and text not in item.text(0).lower())

    def _on_item_clicked(self, item, col):
        data = item.data(0, Qt.UserRole)
        if not data:
            return
        kind, id_ = data
        if kind == "all":
            self.cluster_selected.emit(-1)
        elif kind == "cluster":
            self.cluster_selected.emit(id_)

    def _context_menu(self, pos):
        item = self._tree.itemAt(pos)
        if not item:
            return
        data = item.data(0, Qt.UserRole)
        if not data or data[0] != "cluster":
            return
        cluster_id = data[1]
        cluster_name = item.text(0).split("  (")[0]

        menu = QMenu(self)
        menu.addAction("✨  Propagar etiquetas al grupo (YOLOE)...",
                       lambda: self.cluster_autolabel_requested.emit(cluster_id))
        menu.addAction("🔤  Detectar por texto en el grupo (SAM 3)...",
                       lambda: self.cluster_sam3_requested.emit(cluster_id))
        menu.addSeparator()
        menu.addAction("Marcar como revisado",
                       lambda: self._set_status(cluster_id, "reviewed"))
        menu.addAction("Marcar como descartado",
                       lambda: self._set_status(cluster_id, "discarded"))
        menu.addSeparator()
        act_del = menu.addAction(f"🗑  Eliminar grupo y sus imágenes")
        act_del.setData(cluster_id)
        act_del.triggered.connect(
            lambda: self.cluster_delete_requested.emit(cluster_id)
        )
        menu.exec(self._tree.viewport().mapToGlobal(pos))

    def _set_status(self, cluster_id: int, status: str):
        if self._db is None:
            return
        with self._db.cursor() as cur:
            cur.execute("UPDATE clusters SET status=? WHERE id=?", (status, cluster_id))
        self.refresh()
