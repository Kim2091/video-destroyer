"""Compact desktop launcher for the canonical Video Destroyer workflows."""

import sys
from pathlib import Path

from .gui_config import STAGE_LIBRARY, default_stages, write_temp_create_config


def main():
    try:
        from PySide6.QtCore import QProcess, QTimer, QUrl, Qt
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtWidgets import (
            QAbstractItemView,
            QApplication,
            QCheckBox,
            QComboBox,
            QDoubleSpinBox,
            QFileDialog,
            QFormLayout,
            QFrame,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QListWidget,
            QListWidgetItem,
            QMainWindow,
            QMessageBox,
            QPlainTextEdit,
            QPushButton,
            QStackedWidget,
            QToolButton,
            QVBoxLayout,
            QWidget,
        )
    except ImportError:
        print("The desktop interface requires PySide6. Install it with: python -m pip install '.[gui]'", file=sys.stderr)
        return 2

    class DatasetWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.process = QProcess(self)
            self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
            self.process.readyReadStandardOutput.connect(self._read_output)
            self.process.errorOccurred.connect(self._process_error)
            self.process.finished.connect(self._finished)
            self.current_output = None
            self.generated_config = None
            self.run_buttons = []
            self._moving_codec = False
            self.setWindowTitle("Video Destroyer")
            self.setMinimumSize(860, 570)
            self.resize(1040, 660)
            self._build()

        def _build(self):
            root = QWidget()
            root.setObjectName("root")
            layout = QVBoxLayout(root)
            layout.setContentsMargins(16, 10, 16, 12)
            layout.setSpacing(8)

            toolbar = QFrame()
            toolbar.setObjectName("toolbar")
            toolbar_layout = QHBoxLayout(toolbar)
            toolbar_layout.setContentsMargins(2, 0, 2, 8)
            title = QLabel("Video Destroyer")
            title.setObjectName("title")
            toolbar_layout.addWidget(title)
            toolbar_layout.addStretch()
            self.navigation = {}
            for page_name, label in (("create", "Create"), ("import", "Import"), ("runs", "Runs")):
                button = QPushButton(label)
                button.setObjectName("navButton")
                button.setCheckable(True)
                button.clicked.connect(lambda checked, name=page_name: self._show_page(name))
                self.navigation[page_name] = button
                toolbar_layout.addWidget(button)
            layout.addWidget(toolbar)

            self.workspace = QStackedWidget()
            self.pages = {
                "create": self._create_page(),
                "import": self._import_page(),
                "runs": self._runs_page(),
            }
            for page in self.pages.values():
                self.workspace.addWidget(page)
            self.run_page = self._run_page()
            self.workspace.addWidget(self.run_page)
            layout.addWidget(self.workspace, 1)
            self.return_page = self.pages["create"]
            self._show_page("create")
            self.setCentralWidget(root)

        def _show_page(self, name):
            if self.process.state() != QProcess.ProcessState.NotRunning:
                return
            page = self.pages[name]
            self.workspace.setCurrentWidget(page)
            self.return_page = page
            for page_name, button in self.navigation.items():
                button.setChecked(page_name == name)

        def _create_page(self):
            page = QWidget()
            layout = self._page_layout(page)
            layout.addWidget(self._description("Generate paired clips from a source video or folder."))
            columns = QHBoxLayout()
            columns.setSpacing(16)

            source, form, source_layout = self._form_section("Source and output")
            self.create_input = QLineEdit()
            form.addRow("Source", self._source_picker(self.create_input))
            self.create_output = QLineEdit()
            form.addRow("Run folder", self._folder_picker(self.create_output))
            advanced, advanced_layout = self._advanced_section("Advanced settings")
            self.create_config = QLineEdit()
            advanced_layout.addWidget(QLabel("Optional base v2 configuration"))
            advanced_layout.addWidget(self._file_picker(self.create_config, "YAML files (*.yaml *.yml)"))
            self.create_strict = QCheckBox("Fail the run when items are rejected")
            advanced_layout.addWidget(self.create_strict)
            source_layout.addWidget(advanced)
            columns.addWidget(source, 1)
            columns.addWidget(self._pipeline_editor(), 2)
            layout.addLayout(columns, 1)

            footer = QHBoxLayout()
            footer.addWidget(self._description("Drag stages to reorder the visual pipeline."))
            footer.addStretch()
            footer.addWidget(self._start_button("Create dataset", self._start_create))
            layout.addLayout(footer)
            return page

        def _import_page(self):
            page = QWidget()
            layout = self._page_layout(page)
            layout.addWidget(self._description("Import matching clips without moving or changing source files."))
            columns = QHBoxLayout()
            columns.setSpacing(16)

            clips, form, _ = self._form_section("Paired clips")
            self.import_hr = QLineEdit()
            form.addRow("HR folder", self._folder_picker(self.import_hr))
            self.import_lr = QLineEdit()
            form.addRow("LR folder", self._folder_picker(self.import_lr))
            columns.addWidget(clips, 1)

            options, form, options_layout = self._form_section("Run options")
            self.import_output = QLineEdit()
            form.addRow("Run folder", self._folder_picker(self.import_output))
            self.import_config = QLineEdit()
            form.addRow("Base config", self._file_picker(self.import_config, "YAML files (*.yaml *.yml)"))
            self.materialize = QComboBox()
            self.materialize.addItem("Reference source clips", "")
            self.materialize.addItem("Copy clips into the run", "copy")
            self.materialize.addItem("Hardlink clips into the run", "hardlink")
            form.addRow("Ownership", self.materialize)
            self.import_strict = QCheckBox("Fail the run when items are rejected")
            options_layout.addWidget(self.import_strict)
            columns.addWidget(options, 1)
            layout.addLayout(columns)
            layout.addStretch()

            footer = QHBoxLayout()
            footer.addStretch()
            footer.addWidget(self._start_button("Import and build dataset", self._start_import))
            layout.addLayout(footer)
            return page

        def _runs_page(self):
            page = QWidget()
            layout = self._page_layout(page)
            layout.addWidget(self._description("Continue interrupted work, validate a dataset, or refresh a report."))
            section, form, _ = self._form_section("Existing run")
            self.existing_run = QLineEdit()
            form.addRow("Run folder", self._folder_picker(self.existing_run))
            layout.addWidget(section)
            actions = QHBoxLayout()
            actions.addWidget(self._start_button("Resume", lambda: self._start_existing("resume")))
            actions.addWidget(self._start_button("Validate", lambda: self._start_existing("validate"), primary=False))
            actions.addWidget(self._start_button("Refresh report", lambda: self._start_existing("report"), primary=False))
            actions.addStretch()
            layout.addLayout(actions)
            layout.addStretch()
            return page

        def _run_page(self):
            page = QWidget()
            layout = self._page_layout(page)
            header = QHBoxLayout()
            heading = QLabel("Run progress")
            heading.setObjectName("sectionTitle")
            self.status = QLabel("Ready to start a new run")
            self.status.setObjectName("status")
            header.addWidget(heading)
            header.addStretch()
            header.addWidget(self.status)
            layout.addLayout(header)
            self.log = QPlainTextEdit()
            self.log.setObjectName("log")
            self.log.setReadOnly(True)
            self.log.setMaximumBlockCount(1000)
            self.log.setPlaceholderText("Workflow output will appear here.")
            layout.addWidget(self.log, 1)
            actions = QHBoxLayout()
            self.open_output_button = QPushButton("Open output folder")
            self.open_output_button.setEnabled(False)
            self.open_output_button.clicked.connect(self._open_output)
            self.back_button = QPushButton("Back to setup")
            self.back_button.setEnabled(False)
            self.back_button.clicked.connect(self._return_to_setup)
            self.cancel_button = QPushButton("Cancel run")
            self.cancel_button.setEnabled(False)
            self.cancel_button.clicked.connect(self._cancel)
            actions.addWidget(self.open_output_button)
            actions.addStretch()
            actions.addWidget(self.back_button)
            actions.addWidget(self.cancel_button)
            layout.addLayout(actions)
            return page

        def _return_to_setup(self):
            self.workspace.setCurrentWidget(self.return_page)
            for name, page in self.pages.items():
                self.navigation[name].setChecked(page is self.return_page)

        def _page_layout(self, page):
            layout = QVBoxLayout(page)
            layout.setContentsMargins(4, 6, 4, 4)
            layout.setSpacing(8)
            return layout

        def _description(self, text):
            label = QLabel(text)
            label.setObjectName("description")
            label.setWordWrap(True)
            return label

        def _form_section(self, title):
            section = QFrame()
            section.setObjectName("section")
            layout = QVBoxLayout(section)
            layout.setContentsMargins(12, 10, 12, 10)
            layout.setSpacing(7)
            heading = QLabel(title)
            heading.setObjectName("sectionTitle")
            layout.addWidget(heading)
            form = QFormLayout()
            form.setHorizontalSpacing(12)
            form.setVerticalSpacing(7)
            form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
            layout.addLayout(form)
            return section, form, layout

        def _advanced_section(self, title):
            container = QWidget()
            layout = QVBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(3)
            toggle = QToolButton()
            toggle.setObjectName("advancedToggle")
            toggle.setText(title)
            toggle.setCheckable(True)
            toggle.setArrowType(Qt.ArrowType.RightArrow)
            content = QWidget()
            content.setVisible(False)
            content_layout = QVBoxLayout(content)
            content_layout.setContentsMargins(0, 2, 0, 0)
            content_layout.setSpacing(4)

            def show_content(visible):
                content.setVisible(visible)
                toggle.setArrowType(Qt.ArrowType.DownArrow if visible else Qt.ArrowType.RightArrow)

            toggle.toggled.connect(show_content)
            layout.addWidget(toggle)
            layout.addWidget(content)
            return container, content_layout

        def _pipeline_editor(self):
            section = QFrame()
            section.setObjectName("section")
            layout = QVBoxLayout(section)
            layout.setContentsMargins(12, 10, 12, 10)
            layout.setSpacing(5)
            header = QLabel("Degradation pipeline")
            header.setObjectName("sectionTitle")
            layout.addWidget(header)
            layout.addWidget(self._description("Toggle stages and drag them into the order they should run. Codec encoding is locked as the final step."))
            self.pipeline = QListWidget()
            self.pipeline.setObjectName("pipeline")
            self.pipeline.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
            self.pipeline.setDefaultDropAction(Qt.DropAction.MoveAction)
            self.pipeline.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.pipeline.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.pipeline.setSpacing(1)
            self.pipeline.setFixedHeight(214)
            self.pipeline.model().rowsMoved.connect(self._keep_codec_last)
            for stage in default_stages():
                self._add_pipeline_stage(stage)
            layout.addWidget(self.pipeline)
            return section

        def _add_pipeline_stage(self, stage):
            name = stage["name"]
            definition = STAGE_LIBRARY[name]
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, name)
            if name == "codec":
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsDragEnabled & ~Qt.ItemFlag.ItemIsDropEnabled)
            row = QFrame()
            row.setObjectName("pipelineStage")
            row.setFixedHeight(32)
            layout = QHBoxLayout(row)
            layout.setContentsMargins(8, 2, 8, 2)
            enabled = QCheckBox(definition["title"])
            enabled.setChecked(stage["enabled"])
            enabled.setMinimumWidth(105)
            if name == "codec":
                enabled.setEnabled(False)
            description = QLabel(definition["description"])
            description.setObjectName("pipelineDescription")
            probability = QDoubleSpinBox()
            probability.setDecimals(2)
            probability.setRange(0, 1)
            probability.setSingleStep(0.05)
            probability.setValue(stage["probability"])
            probability.setSuffix(" chance")
            probability.setFixedWidth(108)
            if name == "codec":
                probability.setEnabled(False)
            layout.addWidget(enabled)
            layout.addWidget(description, 1)
            layout.addWidget(probability)
            item.setSizeHint(row.sizeHint())
            self.pipeline.addItem(item)
            self.pipeline.setItemWidget(item, row)

        def _keep_codec_last(self, *_):
            if self._moving_codec:
                return
            for index in range(self.pipeline.count()):
                item = self.pipeline.item(index)
                if item.data(Qt.ItemDataRole.UserRole) == "codec" and index != self.pipeline.count() - 1:
                    self._moving_codec = True
                    widget = self.pipeline.itemWidget(item)
                    self.pipeline.takeItem(index)
                    self.pipeline.addItem(item)
                    self.pipeline.setItemWidget(item, widget)
                    self._moving_codec = False
                    return

        def _pipeline_stages(self):
            stages = []
            for index in range(self.pipeline.count()):
                item = self.pipeline.item(index)
                row = self.pipeline.itemWidget(item)
                stages.append({
                    "name": item.data(Qt.ItemDataRole.UserRole),
                    "enabled": row.findChild(QCheckBox).isChecked(),
                    "probability": row.findChild(QDoubleSpinBox).value(),
                })
            return stages

        def _folder_picker(self, field):
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(field, 1)
            button = QPushButton("Browse")
            button.clicked.connect(lambda: self._choose_folder(field))
            layout.addWidget(button)
            return row

        def _source_picker(self, field):
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(field, 1)
            file_button = QPushButton("File")
            file_button.clicked.connect(lambda: self._choose_file(field, "Video files (*.mp4 *.mkv *.mov *.avi *.webm *.flv *.m4v)"))
            folder_button = QPushButton("Folder")
            folder_button.clicked.connect(lambda: self._choose_folder(field))
            layout.addWidget(file_button)
            layout.addWidget(folder_button)
            return row

        def _file_picker(self, field, filter_text):
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(field, 1)
            button = QPushButton("Browse")
            button.clicked.connect(lambda: self._choose_file(field, filter_text))
            layout.addWidget(button)
            return row

        def _start_button(self, text, callback, primary=True):
            button = QPushButton(text)
            if primary:
                button.setObjectName("primaryButton")
            button.clicked.connect(callback)
            self.run_buttons.append(button)
            return button

        def _choose_folder(self, field):
            selected = QFileDialog.getExistingDirectory(self, "Choose folder", field.text() or str(Path.home()))
            if selected:
                field.setText(selected)

        def _choose_file(self, field, filter_text):
            selected, _ = QFileDialog.getOpenFileName(self, "Choose file", field.text() or str(Path.home()), filter_text)
            if selected:
                field.setText(selected)

        def _start_create(self):
            source, output = self.create_input.text().strip(), self.create_output.text().strip()
            if not self._require_paths(source, output):
                return
            self._cleanup_generated_config()
            try:
                self.generated_config = write_temp_create_config(self._pipeline_stages(), self.create_config.text().strip() or None)
            except ValueError as error:
                QMessageBox.warning(self, "Pipeline configuration", str(error))
                return
            arguments = ["create", source, "--output", output, "--config", str(self.generated_config)]
            if self.create_strict.isChecked():
                arguments.append("--fail-on-rejection")
            self._run(arguments, output)

        def _start_import(self):
            hr, lr, output = self.import_hr.text().strip(), self.import_lr.text().strip(), self.import_output.text().strip()
            if not self._require_paths(hr, lr, output):
                return
            arguments = ["import-pairs", "--hr", hr, "--lr", lr, "--output", output]
            materialize = self.materialize.currentData()
            if materialize:
                arguments.extend(["--materialize", materialize])
            self._append_optional(arguments, self.import_config.text(), self.import_strict.isChecked())
            self._run(arguments, output)

        def _start_existing(self, command):
            run = self.existing_run.text().strip()
            if not self._require_paths(run):
                return
            self._run([command, run], run)

        def _append_optional(self, arguments, config, strict):
            if config.strip():
                arguments.extend(["--config", config.strip()])
            if strict:
                arguments.append("--fail-on-rejection")

        def _require_paths(self, *values):
            if any(not value for value in values):
                QMessageBox.warning(self, "Missing path", "Choose all required input and output paths before starting.")
                return False
            return True

        def _run(self, arguments, output):
            if self.process.state() != QProcess.ProcessState.NotRunning:
                return
            self.current_output = Path(output)
            self.workspace.setCurrentWidget(self.run_page)
            self.log.clear()
            self.log.appendPlainText("$ video-destroyer " + " ".join(arguments))
            self.status.setText("Running workflow...")
            self.open_output_button.setEnabled(False)
            self.back_button.setEnabled(False)
            self.cancel_button.setEnabled(True)
            for button in self.run_buttons:
                button.setEnabled(False)
            for button in self.navigation.values():
                button.setEnabled(False)
            self.process.start(sys.executable, ["-m", "video_destroyer", *arguments])

        def _read_output(self):
            output = bytes(self.process.readAllStandardOutput()).decode(errors="replace")
            if output:
                self.log.appendPlainText(output.rstrip())

        def _process_error(self, error):
            if error == QProcess.ProcessError.FailedToStart:
                self.log.appendPlainText("Unable to start the Video Destroyer process.")
                self.status.setText("Unable to start workflow")
                self._finish_ui()

        def _finished(self, exit_code, _exit_status):
            self._read_output()
            success = exit_code == 0
            self.status.setText("Dataset workflow completed" if success else f"Workflow stopped with exit code {exit_code}")
            if success:
                self.log.appendPlainText("\nFinished successfully. Open the output folder or return to setup.")
            if self.current_output and self.current_output.exists():
                self.open_output_button.setEnabled(True)
            self._finish_ui()

        def _finish_ui(self):
            self.cancel_button.setEnabled(False)
            self.back_button.setEnabled(True)
            for button in self.run_buttons:
                button.setEnabled(True)
            for button in self.navigation.values():
                button.setEnabled(True)
            self._cleanup_generated_config()

        def _cancel(self):
            if self.process.state() == QProcess.ProcessState.NotRunning:
                return
            self.status.setText("Stopping workflow...")
            self.process.terminate()
            QTimer.singleShot(3000, self._kill_if_running)

        def _kill_if_running(self):
            if self.process.state() != QProcess.ProcessState.NotRunning:
                self.process.kill()

        def _open_output(self):
            if self.current_output:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.current_output.resolve())))

        def _cleanup_generated_config(self):
            if self.generated_config:
                self.generated_config.unlink(missing_ok=True)
                self.generated_config = None

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(_stylesheet())
    window = DatasetWindow()
    window.show()
    return app.exec()


def _stylesheet():
    return """
        QWidget#root { background: #121518; color: #e8ebed; font-family: Segoe UI, sans-serif; font-size: 13px; }
        QFrame#toolbar { border-bottom: 1px solid #30363d; }
        QLabel#title { color: #f5f6f7; font-size: 18px; font-weight: 700; }
        QLabel#description { color: #94a0aa; font-size: 12px; }
        QLabel#sectionTitle { color: #eef0f2; font-size: 13px; font-weight: 700; }
        QLabel#status { color: #82c7ae; font-size: 12px; font-weight: 600; }
        QFrame#section { background: #181c20; border: none; border-radius: 5px; }
        QLineEdit, QComboBox, QDoubleSpinBox, QPlainTextEdit { background: #1a1f24; color: #e8ebed; border: 1px solid #343c44; border-radius: 4px; padding: 5px 7px; selection-background-color: #a87535; }
        QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus { border: 1px solid #d8a657; }
        QPlainTextEdit#log { font-family: Cascadia Mono, Consolas, monospace; font-size: 11px; }
        QPushButton { background: #232a30; color: #d9dfe4; border: 1px solid #3b454e; border-radius: 4px; padding: 5px 9px; }
        QPushButton:hover { background: #2d353d; }
        QPushButton:disabled { color: #68737d; background: #1b2025; border-color: #2c343b; }
        QPushButton#navButton { background: transparent; border: none; border-bottom: 2px solid transparent; border-radius: 0; color: #8c97a1; padding: 6px 10px; }
        QPushButton#navButton:checked { color: #f0f2f4; border-bottom-color: #d8a657; }
        QPushButton#primaryButton { background: #d8a657; border-color: #e6bd79; color: #1c1914; font-weight: 700; padding: 7px 13px; }
        QPushButton#primaryButton:hover { background: #e6b96e; }
        QToolButton#advancedToggle { color: #b9c2c9; background: transparent; border: none; padding: 3px 0; }
        QCheckBox { color: #c7cfd6; spacing: 6px; }
        QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #59636d; border-radius: 3px; background: #15191d; }
        QCheckBox::indicator:checked { background: #d8a657; border-color: #e6bd79; }
        QListWidget#pipeline { background: #181c20; border: 1px solid #30363d; border-radius: 4px; padding: 2px; }
        QListWidget#pipeline::item { border: none; }
        QFrame#pipelineStage { background: #1a1f24; border: none; border-bottom: 1px solid #2d353d; }
        QLabel#pipelineDescription { color: #7f8b96; font-size: 11px; }
    """


if __name__ == "__main__":
    raise SystemExit(main())
