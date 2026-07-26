"""Desktop launcher for the canonical Video Destroyer workflows.

The window is a single top-to-bottom flow: pick a workflow, then work down the
numbered steps to the action button in the footer.
"""

import sys
from pathlib import Path

from .gui_config import STAGE_LIBRARY, default_stages, write_temp_create_config


WORKFLOWS = (
    ("create", "Create", "Generate degraded clips from a source video or folder, then extract and validate frames."),
    ("import", "Import", "Pair clips you already have. Source files are read in place and never modified."),
    ("runs", "Existing run", "Resume interrupted work, re-validate a dataset, or rewrite its report."),
)


def main():
    try:
        from PySide6.QtCore import QProcess, QTimer, QUrl, Qt
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtWidgets import (
            QAbstractItemView,
            QApplication,
            QButtonGroup,
            QCheckBox,
            QComboBox,
            QDoubleSpinBox,
            QFileDialog,
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
            QRadioButton,
            QScrollArea,
            QSizePolicy,
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
            self.stage_rows = {}
            self._moving_codec = False
            self.setWindowTitle("Video Destroyer")
            self.setMinimumSize(880, 600)
            self.resize(1020, 820)
            self._build()

        # ------------------------------------------------------------------ shell

        def _build(self):
            root = QWidget()
            root.setObjectName("root")
            layout = QVBoxLayout(root)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            layout.addWidget(self._header())

            self.workspace = QStackedWidget()
            self.pages = {}
            builders = {"create": self._create_page, "import": self._import_page, "runs": self._runs_page}
            for name, _, description in WORKFLOWS:
                page = builders[name](description)
                self.pages[name] = page
                self.workspace.addWidget(page)
            self.run_page = self._run_page()
            self.workspace.addWidget(self.run_page)
            layout.addWidget(self.workspace, 1)

            self.return_page = self.pages["create"]
            self._show_page("create")
            self.setCentralWidget(root)

        def _header(self):
            header = QFrame()
            header.setObjectName("header")
            layout = QHBoxLayout(header)
            layout.setContentsMargins(24, 14, 24, 14)
            layout.setSpacing(12)
            title = QLabel("VIDEO DESTROYER")
            title.setObjectName("wordmark")
            layout.addWidget(title)
            layout.addStretch(1)

            switcher = QFrame()
            switcher.setObjectName("switcher")
            switcher_layout = QHBoxLayout(switcher)
            switcher_layout.setContentsMargins(3, 3, 3, 3)
            switcher_layout.setSpacing(2)
            self.navigation = {}
            for name, label, _ in WORKFLOWS:
                button = QPushButton(label)
                button.setObjectName("modeButton")
                button.setCheckable(True)
                button.setCursor(Qt.CursorShape.PointingHandCursor)
                button.clicked.connect(lambda _checked, page=name: self._show_page(page))
                self.navigation[name] = button
                switcher_layout.addWidget(button)
            layout.addWidget(switcher)
            return header

        def _show_page(self, name):
            if self.process.state() != QProcess.ProcessState.NotRunning:
                return
            page = self.pages[name]
            self.workspace.setCurrentWidget(page)
            self.return_page = page
            for page_name, button in self.navigation.items():
                button.setChecked(page_name == name)

        def _scaffold(self, description):
            """Return (page, steps_layout, footer_layout) for a scrollable flow page."""
            page = QWidget()
            outer = QVBoxLayout(page)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.setSpacing(0)

            scroll = QScrollArea()
            scroll.setObjectName("scroll")
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            body = QWidget()
            body.setObjectName("flow")
            steps = QVBoxLayout(body)
            steps.setContentsMargins(24, 14, 24, 14)
            steps.setSpacing(0)
            intro = QLabel(description)
            intro.setObjectName("intro")
            intro.setWordWrap(True)
            steps.addWidget(intro)
            steps.addSpacing(10)
            scroll.setWidget(body)
            outer.addWidget(scroll, 1)

            footer = QFrame()
            footer.setObjectName("footer")
            footer_layout = QHBoxLayout(footer)
            footer_layout.setContentsMargins(24, 12, 24, 12)
            footer_layout.setSpacing(12)
            outer.addWidget(footer)
            return page, steps, footer_layout

        def _step(self, layout, number, title, hint=""):
            """Add a numbered step card and return the layout for its contents."""
            if number > 1:
                arrow = QLabel("↓")
                arrow.setObjectName("flowArrow")
                arrow.setFixedWidth(52)
                arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(arrow)

            card = QFrame()
            card.setObjectName("step")
            outer = QHBoxLayout(card)
            outer.setContentsMargins(14, 12, 14, 14)
            outer.setSpacing(12)

            gutter = QVBoxLayout()
            gutter.setContentsMargins(0, 0, 0, 0)
            badge = QLabel(str(number))
            badge.setObjectName("stepNumber")
            badge.setFixedSize(24, 24)
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            gutter.addWidget(badge)
            gutter.addStretch(1)
            outer.addLayout(gutter)

            body = QVBoxLayout()
            body.setContentsMargins(0, 0, 0, 0)
            body.setSpacing(8)
            heading = QLabel(title)
            heading.setObjectName("stepTitle")
            heading.setFixedHeight(24)
            body.addWidget(heading)
            if hint:
                body.addWidget(self._hint(hint))
            outer.addLayout(body, 1)
            layout.addWidget(card)
            return body

        # ------------------------------------------------------------------ pages

        def _create_page(self, description):
            page, steps, footer = self._scaffold(description)

            source = self._step(steps, 1, "Source", "The video file or folder of videos to degrade.")
            self.create_input = QLineEdit()
            self.create_input.setPlaceholderText("Choose a video file or a folder of videos")
            source.addWidget(self._source_picker(self.create_input))

            pipeline = self._step(steps, 2, "Degradations")
            pipeline.addWidget(self._pipeline_editor())

            output = self._step(steps, 3, "Output", "A new run folder for the clips, frames, reports, and logs.")
            self.create_output = QLineEdit()
            self.create_output.setPlaceholderText("Choose an empty run folder")
            output.addWidget(self._folder_picker(self.create_output))
            advanced, advanced_layout = self._advanced_section()
            self.create_config = QLineEdit()
            self.create_config.setPlaceholderText("Optional base configuration (version 2 YAML)")
            advanced_layout.addWidget(self._file_picker(self.create_config, "YAML files (*.yaml *.yml)"))
            self.create_strict = QCheckBox("Fail the run when items are rejected")
            advanced_layout.addWidget(self.create_strict)
            output.addWidget(advanced)

            steps.addStretch(1)
            footer.addWidget(self._hint("Nothing is published until validation passes.", wrap=False))
            footer.addStretch(1)
            footer.addWidget(self._start_button("Create dataset  →", self._start_create))
            return page

        def _import_page(self, description):
            page, steps, footer = self._scaffold(description)

            clips = self._step(steps, 1, "Paired clips", "Matched by relative path without the final extension.")
            self.import_hr = QLineEdit()
            self.import_hr.setPlaceholderText("Choose the HR clip folder")
            clips.addWidget(self._captioned("HR clips", self._folder_picker(self.import_hr)))
            self.import_lr = QLineEdit()
            self.import_lr.setPlaceholderText("Choose the LR clip folder")
            clips.addWidget(self._captioned("LR clips", self._folder_picker(self.import_lr)))

            handling = self._step(steps, 2, "File handling", "How the run should take ownership of those clips.")
            self.materialize = QComboBox()
            self.materialize.addItem("Reference the source clips in place", "")
            self.materialize.addItem("Copy the clips into the run", "copy")
            self.materialize.addItem("Hardlink the clips into the run", "hardlink")
            handling.addWidget(self.materialize)

            output = self._step(steps, 3, "Output", "A new run folder for the frames, reports, and logs.")
            self.import_output = QLineEdit()
            self.import_output.setPlaceholderText("Choose an empty run folder")
            output.addWidget(self._folder_picker(self.import_output))
            advanced, advanced_layout = self._advanced_section()
            self.import_config = QLineEdit()
            self.import_config.setPlaceholderText("Optional base configuration (version 2 YAML)")
            advanced_layout.addWidget(self._file_picker(self.import_config, "YAML files (*.yaml *.yml)"))
            self.import_strict = QCheckBox("Fail the run when items are rejected")
            advanced_layout.addWidget(self.import_strict)
            output.addWidget(advanced)

            steps.addStretch(1)
            footer.addWidget(self._hint("Source clips are only ever read.", wrap=False))
            footer.addStretch(1)
            footer.addWidget(self._start_button("Import and build  →", self._start_import))
            return page

        def _runs_page(self, description):
            page, steps, footer = self._scaffold(description)

            target = self._step(steps, 1, "Run folder", "A folder created by an earlier create or import run.")
            self.existing_run = QLineEdit()
            self.existing_run.setPlaceholderText("Choose an existing run folder")
            target.addWidget(self._folder_picker(self.existing_run))

            action = self._step(steps, 2, "Action", "Pick what to do with that run.")
            self.run_action = QButtonGroup(page)
            for index, (label, command, hint) in enumerate((
                ("Resume", "resume", "Continue an interrupted run from its saved state."),
                ("Validate", "validate", "Re-check the published dataset without changing it."),
                ("Refresh report", "report", "Rewrite the run summary from existing manifests."),
            )):
                choice = QRadioButton(label)
                choice.setMinimumWidth(140)
                choice.setChecked(index == 0)
                choice.setProperty("command", command)
                self.run_action.addButton(choice, index)
                row = QWidget()
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(10)
                row_layout.addWidget(choice)
                row_layout.addWidget(self._hint(hint), 1)
                action.addWidget(row)
            self.run_action.buttonToggled.connect(self._sync_action_button)

            steps.addStretch(1)
            footer.addWidget(self._hint("Each action reuses the run's immutable configuration.", wrap=False))
            footer.addStretch(1)
            self.existing_button = self._start_button("Resume  →", self._start_selected_action)
            footer.addWidget(self.existing_button)
            return page

        def _sync_action_button(self, button, checked):
            if checked:
                self.existing_button.setText(f"{button.text()}  →")

        def _start_selected_action(self):
            self._start_existing(self.run_action.checkedButton().property("command"))

        def _run_page(self):
            page = QWidget()
            outer = QVBoxLayout(page)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.setSpacing(0)

            body = QWidget()
            body.setObjectName("flow")
            layout = QVBoxLayout(body)
            layout.setContentsMargins(24, 18, 24, 18)
            layout.setSpacing(10)

            header = QHBoxLayout()
            heading = QLabel("Run progress")
            heading.setObjectName("stepTitle")
            self.status = QLabel("Ready to start a new run")
            self.status.setObjectName("status")
            header.addWidget(heading)
            header.addStretch(1)
            header.addWidget(self.status)
            layout.addLayout(header)

            self.log = QPlainTextEdit()
            self.log.setObjectName("log")
            self.log.setReadOnly(True)
            self.log.setMaximumBlockCount(2000)
            self.log.setPlaceholderText("Workflow output will appear here.")
            layout.addWidget(self.log, 1)
            outer.addWidget(body, 1)

            footer = QFrame()
            footer.setObjectName("footer")
            actions = QHBoxLayout(footer)
            actions.setContentsMargins(24, 12, 24, 12)
            actions.setSpacing(12)
            self.open_output_button = QPushButton("Open output folder")
            self.open_output_button.setEnabled(False)
            self.open_output_button.clicked.connect(self._open_output)
            self.back_button = QPushButton("←  Back to setup")
            self.back_button.setEnabled(False)
            self.back_button.clicked.connect(self._return_to_setup)
            self.cancel_button = QPushButton("Cancel run")
            self.cancel_button.setObjectName("dangerButton")
            self.cancel_button.setEnabled(False)
            self.cancel_button.clicked.connect(self._cancel)
            actions.addWidget(self.back_button)
            actions.addStretch(1)
            actions.addWidget(self.open_output_button)
            actions.addWidget(self.cancel_button)
            outer.addWidget(footer)
            return page

        def _return_to_setup(self):
            self.workspace.setCurrentWidget(self.return_page)
            for name, page in self.pages.items():
                self.navigation[name].setChecked(page is self.return_page)

        # --------------------------------------------------------- pipeline editor

        def _pipeline_editor(self):
            container = QWidget()
            layout = QVBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(6)

            columns = QHBoxLayout()
            columns.setContentsMargins(9, 0, 9, 0)
            columns.setSpacing(8)
            stage_caption = QLabel("Stage — drag to reorder, runs top to bottom")
            stage_caption.setObjectName("caption")
            stage_caption.setIndent(38)
            chance_caption = QLabel("Chance")
            chance_caption.setObjectName("caption")
            chance_caption.setFixedWidth(74)
            chance_caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
            columns.addWidget(stage_caption, 1)
            columns.addWidget(chance_caption)
            layout.addLayout(columns)

            self.pipeline_summary = QLabel()
            self.pipeline_summary.setObjectName("hint")
            self.pipeline = QListWidget()
            self.pipeline.setObjectName("pipeline")
            self.pipeline.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
            self.pipeline.setDefaultDropAction(Qt.DropAction.MoveAction)
            self.pipeline.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            self.pipeline.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.pipeline.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.pipeline.setSpacing(2)
            self.pipeline.model().rowsMoved.connect(self._keep_codec_last)
            for stage in default_stages():
                self._add_pipeline_stage(stage)
            self._fit_pipeline_height()
            layout.addWidget(self.pipeline)
            layout.addWidget(self.pipeline_summary)
            self._renumber_pipeline()
            return container

        def _add_pipeline_stage(self, stage):
            name = stage["name"]
            definition = STAGE_LIBRARY[name]
            locked = name == "codec"

            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, name)
            if locked:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsDragEnabled & ~Qt.ItemFlag.ItemIsDropEnabled)

            row = QFrame()
            row.setObjectName("stageRowLocked" if locked else "stageRow")
            row.setFixedHeight(34)
            layout = QHBoxLayout(row)
            layout.setContentsMargins(8, 0, 8, 0)
            layout.setSpacing(8)

            badge = QLabel()
            badge.setObjectName("stageNumber")
            badge.setFixedSize(20, 20)
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            handle = QLabel("" if locked else "≡")
            handle.setObjectName("stageHandle")
            handle.setFixedWidth(10)
            handle.setToolTip("Always runs last" if locked else "Drag to reorder")

            enabled = QCheckBox(definition["title"])
            enabled.setChecked(stage["enabled"])
            enabled.setMinimumWidth(124)
            enabled.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
            description = QLabel(definition["description"])
            description.setObjectName("stageDescription")
            probability = QDoubleSpinBox()
            probability.setDecimals(2)
            probability.setRange(0, 1)
            probability.setSingleStep(0.05)
            probability.setValue(stage["probability"])
            probability.setFixedWidth(74)
            probability.setAlignment(Qt.AlignmentFlag.AlignCenter)
            if locked:
                enabled.setEnabled(False)
                probability.setEnabled(False)
                description.setText("Required final step")

            layout.addWidget(badge)
            layout.addWidget(handle)
            layout.addWidget(enabled)
            layout.addWidget(description, 1)
            layout.addWidget(probability)

            self.stage_rows[name] = {"row": row, "badge": badge, "enabled": enabled, "probability": probability}
            enabled.toggled.connect(lambda checked, stage_name=name: self._stage_toggled(stage_name, checked))
            item.setSizeHint(row.sizeHint())
            self.pipeline.addItem(item)
            self.pipeline.setItemWidget(item, row)
            self._stage_toggled(name, enabled.isChecked())

        def _stage_toggled(self, name, checked):
            widgets = self.stage_rows[name]
            for widget in (widgets["badge"], widgets["row"], widgets["probability"]):
                widget.setProperty("off", not checked)
                widget.style().unpolish(widget)
                widget.style().polish(widget)
            self._update_pipeline_summary()

        def _keep_codec_last(self, *_):
            if self._moving_codec:
                return
            last = self.pipeline.count() - 1
            for index in range(self.pipeline.count()):
                item = self.pipeline.item(index)
                if item.data(Qt.ItemDataRole.UserRole) == "codec" and index != last:
                    self._moving_codec = True
                    self.pipeline.removeItemWidget(item)
                    self.pipeline.takeItem(index)
                    self._add_pipeline_stage({"name": "codec", "enabled": True, "probability": 1.0})
                    self._moving_codec = False
                    break
            self._renumber_pipeline()

        def _fit_pipeline_height(self):
            last = self.pipeline.item(self.pipeline.count() - 1)
            self.pipeline.setFixedHeight(self.pipeline.visualItemRect(last).bottom() + 8)

        def _renumber_pipeline(self):
            for index in range(self.pipeline.count()):
                name = self.pipeline.item(index).data(Qt.ItemDataRole.UserRole)
                self.stage_rows[name]["badge"].setText(str(index + 1))
            self._update_pipeline_summary()

        def _update_pipeline_summary(self):
            stages = self._pipeline_stages()
            active = [stage for stage in stages if stage["enabled"]]
            order = " → ".join(STAGE_LIBRARY[stage["name"]]["title"] for stage in active)
            self.pipeline_summary.setText(f"{len(active)} of {len(stages)} active:  {order}")

        def _pipeline_stages(self):
            stages = []
            for index in range(self.pipeline.count()):
                name = self.pipeline.item(index).data(Qt.ItemDataRole.UserRole)
                widgets = self.stage_rows[name]
                stages.append({
                    "name": name,
                    "enabled": widgets["enabled"].isChecked(),
                    "probability": widgets["probability"].value(),
                })
            return stages

        # ------------------------------------------------------------- small parts

        def _hint(self, text, wrap=True):
            label = QLabel(text)
            label.setObjectName("hint")
            label.setWordWrap(wrap)
            return label

        def _captioned(self, text, widget):
            container = QWidget()
            layout = QVBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(3)
            caption = QLabel(text)
            caption.setObjectName("caption")
            layout.addWidget(caption)
            layout.addWidget(widget)
            return container

        def _advanced_section(self):
            container = QWidget()
            layout = QVBoxLayout(container)
            layout.setContentsMargins(0, 2, 0, 0)
            layout.setSpacing(6)
            toggle = QToolButton()
            toggle.setObjectName("advancedToggle")
            toggle.setText("▸  Advanced")
            toggle.setCheckable(True)
            toggle.setCursor(Qt.CursorShape.PointingHandCursor)
            content = QWidget()
            content.setVisible(False)
            content_layout = QVBoxLayout(content)
            content_layout.setContentsMargins(0, 0, 0, 0)
            content_layout.setSpacing(7)

            def show_content(visible):
                content.setVisible(visible)
                toggle.setText(("▾  " if visible else "▸  ") + "Advanced")

            toggle.toggled.connect(show_content)
            layout.addWidget(toggle, 0, Qt.AlignmentFlag.AlignLeft)
            layout.addWidget(content)
            return container, content_layout

        def _picker_row(self, field, buttons):
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(6)
            layout.addWidget(field, 1)
            for text, callback in buttons:
                button = QPushButton(text)
                button.setFixedWidth(76)
                button.clicked.connect(callback)
                layout.addWidget(button)
            return row

        def _folder_picker(self, field):
            return self._picker_row(field, [("Browse", lambda: self._choose_folder(field))])

        def _source_picker(self, field):
            video_filter = "Video files (*.mp4 *.mkv *.mov *.avi *.webm *.flv *.m4v)"
            return self._picker_row(field, [
                ("File", lambda: self._choose_file(field, video_filter)),
                ("Folder", lambda: self._choose_folder(field)),
            ])

        def _file_picker(self, field, filter_text):
            return self._picker_row(field, [("Browse", lambda: self._choose_file(field, filter_text))])

        def _start_button(self, text, callback, primary=True):
            button = QPushButton(text)
            button.setObjectName("primaryButton" if primary else "secondaryButton")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
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

        # ------------------------------------------------------------------- runs

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
            self._set_status("Running workflow…", "busy")
            self.open_output_button.setEnabled(False)
            self.back_button.setEnabled(False)
            self.cancel_button.setEnabled(True)
            for button in self.run_buttons:
                button.setEnabled(False)
            for button in self.navigation.values():
                button.setEnabled(False)
            self.process.start(sys.executable, ["-m", "video_destroyer", *arguments])

        def _set_status(self, text, state):
            self.status.setText(text)
            self.status.setProperty("state", state)
            self.status.style().unpolish(self.status)
            self.status.style().polish(self.status)

        def _read_output(self):
            output = bytes(self.process.readAllStandardOutput()).decode(errors="replace")
            if output:
                self.log.appendPlainText(output.rstrip())

        def _process_error(self, error):
            if error == QProcess.ProcessError.FailedToStart:
                self.log.appendPlainText("Unable to start the Video Destroyer process.")
                self._set_status("Unable to start workflow", "error")
                self._finish_ui()

        def _finished(self, exit_code, _exit_status):
            self._read_output()
            success = exit_code == 0
            if success:
                self._set_status("Completed", "ok")
                self.log.appendPlainText("\nFinished successfully. Open the output folder or return to setup.")
            else:
                self._set_status(f"Stopped with exit code {exit_code}", "error")
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
            self._set_status("Stopping workflow…", "busy")
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
        QWidget { font-family: Segoe UI, sans-serif; font-size: 13px; }
        QWidget#root { background: #0f1215; color: #e6eaee; }
        QWidget#flow { background: #0f1215; }
        QScrollArea#scroll { background: #0f1215; border: none; }

        QFrame#header { background: #14181c; border-bottom: 1px solid #232a31; }
        QLabel#wordmark { color: #f2f5f7; font-size: 13px; font-weight: 700; letter-spacing: 2px; }
        QFrame#switcher { background: #1a1f25; border: 1px solid #262d34; border-radius: 7px; }
        QPushButton#modeButton { background: transparent; border: none; border-radius: 5px; color: #8d99a4; padding: 6px 16px; font-weight: 600; }
        QPushButton#modeButton:hover { color: #d3dae0; }
        QPushButton#modeButton:checked { background: #d8a657; color: #1a1710; }
        QPushButton#modeButton:disabled { color: #5a646d; }
        QPushButton#modeButton:checked:disabled { background: #6f5730; color: #221d14; }

        QLabel#intro { color: #8d99a4; font-size: 12px; }
        QLabel#hint { color: #6f7a85; font-size: 11px; }
        QLabel#caption { color: #8d99a4; font-size: 11px; font-weight: 600; letter-spacing: 0.4px; }
        QLabel#flowArrow { color: #4a545e; font-size: 15px; padding: 0; }

        QFrame#step { background: #171b20; border: 1px solid #232a31; border-radius: 8px; }
        QLabel#stepNumber { background: #262d34; color: #d8a657; border-radius: 12px; font-size: 12px; font-weight: 700; }
        QLabel#stepTitle { color: #f2f5f7; font-size: 14px; font-weight: 700; }

        QLineEdit, QComboBox, QDoubleSpinBox, QPlainTextEdit { background: #11151a; color: #e6eaee; border: 1px solid #2b333b; border-radius: 6px; padding: 6px 9px; selection-background-color: #a87535; }
        QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus { border-color: #d8a657; }
        QLineEdit:hover, QComboBox:hover { border-color: #3a444e; }
        QComboBox::drop-down { border: none; width: 22px; }
        QComboBox QAbstractItemView { background: #171b20; color: #e6eaee; border: 1px solid #2b333b; selection-background-color: #d8a657; selection-color: #1a1710; outline: none; }
        QPlainTextEdit#log { font-family: Cascadia Mono, Consolas, monospace; font-size: 11px; padding: 10px; }

        QPushButton { background: #232a31; color: #d3dae0; border: 1px solid #333d46; border-radius: 6px; padding: 6px 12px; }
        QPushButton:hover { background: #2c343d; border-color: #43505b; }
        QPushButton:pressed { background: #1e252b; }
        QPushButton:disabled { color: #5a646d; background: #191e23; border-color: #262d34; }
        QPushButton#primaryButton { background: #d8a657; border-color: #e6bd79; color: #1a1710; font-weight: 700; padding: 8px 18px; }
        QPushButton#primaryButton:hover { background: #e6bd79; }
        QPushButton#primaryButton:disabled { background: #6f5730; border-color: #6f5730; color: #2b2419; }
        QPushButton#secondaryButton { padding: 8px 18px; font-weight: 600; }
        QPushButton#dangerButton { color: #e08b7d; border-color: #4a3330; }
        QPushButton#dangerButton:hover { background: #3a2724; border-color: #6b4640; }
        QToolButton#advancedToggle { color: #8d99a4; background: transparent; border: none; padding: 2px 0; font-size: 12px; }
        QToolButton#advancedToggle:hover { color: #d3dae0; }

        QFrame#footer { background: #14181c; border-top: 1px solid #232a31; }
        QLabel#status { font-size: 12px; font-weight: 600; color: #8d99a4; }
        QLabel#status[state="busy"] { color: #d8a657; }
        QLabel#status[state="ok"] { color: #7fb69a; }
        QLabel#status[state="error"] { color: #e08b7d; }

        QCheckBox { color: #d3dae0; spacing: 7px; }
        QCheckBox:disabled { color: #8d99a4; }
        QCheckBox::indicator { width: 15px; height: 15px; border: 1px solid #4a545e; border-radius: 4px; background: #11151a; }
        QCheckBox::indicator:hover { border-color: #d8a657; }
        QCheckBox::indicator:checked { background: #d8a657; border-color: #e6bd79; }
        QCheckBox::indicator:checked:disabled { background: #6f5730; border-color: #6f5730; }

        QRadioButton { color: #d3dae0; spacing: 7px; }
        QRadioButton::indicator { width: 15px; height: 15px; border: 1px solid #4a545e; border-radius: 8px; background: #11151a; }
        QRadioButton::indicator:hover { border-color: #d8a657; }
        QRadioButton::indicator:checked { background: #d8a657; border-color: #e6bd79; }

        QListWidget#pipeline { background: #11151a; border: 1px solid #2b333b; border-radius: 6px; padding: 3px; outline: none; }
        QListWidget#pipeline::item { border: none; border-radius: 5px; }
        QListWidget#pipeline::item:selected { background: transparent; }
        QFrame#stageRow { background: #1a1f25; border: 1px solid #262d34; border-radius: 5px; }
        QFrame#stageRow[off="true"] { background: #14181c; border-color: #1f252b; }
        QFrame#stageRowLocked { background: #1a1f25; border: 1px solid #262d34; border-left: 2px solid #d8a657; border-radius: 5px; }
        QLabel#stageNumber { background: #2b333b; color: #d8a657; border-radius: 10px; font-size: 11px; font-weight: 700; }
        QLabel#stageNumber[off="true"] { background: #1f252b; color: #5a646d; }
        QLabel#stageHandle { color: #5c6873; font-size: 14px; }
        QLabel#stageDescription { color: #6f7a85; font-size: 11px; }
        QDoubleSpinBox[off="true"] { color: #5a646d; }
        QDoubleSpinBox { padding: 3px 4px; }
        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 14px; background: transparent; border: none; }

        QScrollBar:vertical { background: transparent; width: 9px; margin: 0; }
        QScrollBar::handle:vertical { background: #2f3841; border-radius: 4px; min-height: 30px; }
        QScrollBar::handle:vertical:hover { background: #3d4852; }
        QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page { background: none; border: none; height: 0; }
    """


if __name__ == "__main__":
    raise SystemExit(main())
