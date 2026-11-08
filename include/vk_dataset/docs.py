"""
What the app tells a person about each node: its name, a one-line summary for
the node card, a longer help text, where it goes in a job, and what each of
its ports means.

All of the wording lives here, so it can be read and edited in one place and
kept consistent; registry.py merges it into each node's declaration.
test_docs.py fails when a node type has no entry, so a new node cannot ship
undocumented.

Stages give the intended order of a job, left to right. A node may go
anywhere its ports allow, but a job reads best, and usually works best, when
its nodes follow the stages. `also` lists other stages a node is commonly
used in, such as Resample shrinking a large source during Prepare. Nodes of
one `chain` are the devices of one signal path; the palette keeps them
together in the order they are listed here.

Value nodes have no stage: they feed settings, not the video, so they go
wherever the settings they feed are. They are listed under their `group`.
"""

STAGES = [
    (1, "Load", "Bring in the video, or an existing dataset, to work from."),
    (2, "Prepare", "Fix the source once, before it is cut up: tonemap HDR, crop, resize, or line up two "
                   "releases of the same footage."),
    (3, "Split", "Cut the video into chunks. Every node after this runs once per chunk and draws its own "
                 "random settings, so each chunk is degraded differently."),
    (4, "Degrade", "Damage the LR copy the way real low-quality video is damaged: smaller, blurrier, noisier, "
                   "compressed."),
    (5, "Clean up", "Improve the HR target: denoise, dehalo, fix colour or gamma."),
    (6, "Pick", "Choose which runs of frames become training sequences."),
    (7, "Filter", "Throw out sequences that would teach the model the wrong thing, or cut them into tiles."),
    (8, "Save", "Write the dataset to disk."),
]

STAGE_NAMES = {number: name for number, name, _ in STAGES}

# Port meanings shared by many nodes.
VIDEO_IN = ("Video", "The video to work on.")
VIDEO_OUT = ("Video", "The result.")
HR = ("HR", "The high-quality picture: what the model should learn to produce.")
LR = ("LR", "The low-quality picture: what the model is given as input.")
SEQUENCES = ("Sequences", "The picked runs of frames, with any crops, on their way to being saved.")
SIGNAL = ("Signal", "The analogue signal so far: the chain's devices and settings, simulated at Decode.")
VALUE = ("Value", "Wire it into any setting's dot, on the left of its row, to drive that setting.")

NODE_DOCS = {
    # 1 Load
    "load_video": dict(
        stage=1, label="Load Video",
        summary="Opens a video file, or every video in a folder one after another.",
        help="Start every job that works from video with this. Point it at one file, or at a folder to "
             "process each video in it as a batch. For pairs made from two releases, use two Load Video "
             "nodes: one for the HR release and one for the LR release. Folders are then paired by sorted "
             "file name.",
        ports={"clip": ("Video", "The whole video, frame for frame.")}),
    "load_dataset": dict(
        stage=1, label="Load Existing Dataset",
        summary="Reads a dataset folder made earlier (hr/ and lr/ with showN_FrameM images).",
        help="Use this to clean up the HR targets of a dataset you already made, without touching its "
             "original files. Connect its outputs through the Clean up nodes into Save Cleaned-up Dataset.",
        ports={"hr": ("HR targets", "One HR target image per sequence."),
               "lr": ("LR centre frames", "The LR frame matching each HR target."),
               "lr_all": ("All LR frames", "Every LR frame of every sequence, for nodes that change LR too.")}),

    # 2 Prepare
    "tonemap": dict(
        stage=2, label="HDR to SDR",
        summary="Converts HDR video (PQ or HLG) to ordinary SDR. SDR video passes through untouched.",
        help="Put this right after Load Video when the source may be HDR, so both HR and LR are SDR. With "
             "auto detect on, SDR sources are left alone, so it is safe to leave in a job.",
        ports={"clip": VIDEO_IN}),
    "crop_rescale": dict(
        stage=2, label="Crop and Rescale",
        summary="Trims black bars or edges off the frame, then optionally rescales it.",
        help="Use it to remove letterboxing or overscan before anything else sees the picture. In a pair "
             "job, give HR and LR their own Crop and Rescale so each release can be trimmed differently.",
        ports={"clip": VIDEO_IN}),
    "align_time": dict(
        stage=2, label="Sync LR to HR (Timing)",
        summary="Finds which LR frame shows each HR frame, and retimes LR to match HR frame for frame.",
        help="For pairs made from two releases. Two releases rarely line up: one may start later, run at a "
             "different speed (PAL), or have scenes cut. This finds the mapping and returns LR with one "
             "frame per HR frame. Put it after any cropping and before Align HR to LR (Position).",
        ports={"hr": ("HR", "The HR release."), "lr": ("LR", "The LR release, to be retimed.")},
        outputs={"lr": ("LR, synced", "LR with one frame per HR frame.")}),
    "align_space": dict(
        stage=2, label="Align HR to LR (Position)",
        summary="Warps the HR picture so every detail sits exactly over the same detail in LR.",
        help="For pairs made from two releases, after Sync LR to HR (Timing). Releases are framed and "
             "scaled slightly differently; training on misaligned pairs teaches the model to blur. Set "
             "the scale to the model's upscale factor.",
        ports={"hr": ("HR", "The HR release, to be warped."), "lr": ("LR", "The synced LR release.")}),

    # 3 Split
    "chunk": dict(
        stage=3, label="Split into Chunks",
        summary="Cuts the video into chunks by scene, length or frame count. Randomness is per chunk from here.",
        help="Every job made from video needs one. Each chunk gets its own random degradation settings, so "
             "splitting by scene gives every shot a different kind of damage. Connect the clean video to "
             "'Video'; in a pair job, connect LR to 'Second video' so both are cut at the same frames.",
        ports={"clip": ("Video", "The video to split. Scene detection looks at this one."),
               "aux": ("Second video", "Optional. Cut at exactly the same frames, e.g. the synced LR release.")},
        outputs={"clip": ("Video chunk", "The current chunk of the video."),
                 "aux": ("Second video chunk", "The same chunk of the second video.")}),
    "find_static_runs": dict(
        stage=3, label="Find Still Shots",
        summary="Finds runs of frames where the picture does not move, for building a denoise dataset.",
        help="The first step of a denoise dataset. Averaging the frames of a still shot removes grain and "
             "noise but keeps the picture, which gives a clean HR target for the noisy original. Small "
             "film gate weave is tolerated.",
        ports={"clip": VIDEO_IN}, outputs={"runs": ("Still shots", "The frame ranges of each still shot.")}),

    # 4 Degrade
    "optics_blur": dict(
        stage=4, label="Lens Blur",
        summary="Softens the picture like a lens or camera, with a random blur shape per chunk.",
        help="Round, stretched or flat-topped blur shapes (the Real-ESRGAN family), in linear light. Put it "
             "first, at HR size. Keep it mild: heavy blur teaches a model to smear.",
        ports={"clip": VIDEO_IN}),
    "film_grain": dict(
        stage=4, label="Film Grain",
        summary="Adds film grain: clumped, strongest in mid-tones, new on every frame.",
        help="On the LR side only, the model learns to remove grain. Before the HR/LR split, both get it "
             "and the model learns to keep it.",
        ports={"clip": VIDEO_IN}),
    "sensor_noise": dict(
        stage=4, label="Sensor Noise",
        summary="Adds digital camera noise: stronger in shadows, new on every frame.",
        help="For phone, webcam and low-light footage. Put it early, before scaling and compression.",
        ports={"clip": VIDEO_IN}),
    "resample": dict(
        stage=4, also=[2], label="Resample",
        summary="Scales to a DVD frame or by a fraction, with a realistic random scaler per chunk.",
        help="Use it where the real chain changed size: mastering down to SD, a player scaling up. DVD "
             "sizes are anamorphic, as on a disc; Scale to Model Size fixes the shape at the end.",
        ports={"clip": VIDEO_IN}),
    "sharpen": dict(
        stage=4, label="Sharpen (Halos)",
        summary="Sharpens edges with an unsharp mask, leaving the halos of DVDs and broadcasts.",
        help="Mastering sharpening goes after the SD resample; horizontal-only is the analogue kind.",
        ports={"clip": VIDEO_IN}),
    "chroma_bleed": dict(
        stage=4, label="Chroma Bleed",
        summary="Blurs colour sideways and shifts it off the lines, as analogue video does.",
        help="Keep it mild for DVD; stronger for VHS and old broadcasts.",
        ports={"clip": VIDEO_IN}),
    "telecine": dict(
        stage=4, label="Telecine",
        summary="Interlaces the picture as a studio does for NTSC: 3:2 pulldown of film, or native video.",
        help="'film' turns every 4 frames into 5, two of them mixing fields. 'video' is native interlaced "
             "video, the frame count kept. Put it at SD size; PAL discs (720x576) have no pulldown. A Bad "
             "Telecine Fix must follow, usually after Compression: until then LR does not line up with HR.",
        ports={"clip": VIDEO_IN}),
    "telecine_fix": dict(
        stage=4, label="Bad Telecine Fix",
        summary="Turns the fields back into one frame per HR frame, badly: combing, blended or jaggy frames.",
        help="What a player or ripper does with a telecined disc. Field matching fails on some frames and "
             "keeps a mixed one; the recovery is what the deinterlacer then does. Mixed frames 0 with "
             "'combed' gives a clean recovery. Chunks the Telecine skipped pass through.",
        ports={"clip": VIDEO_IN}),
    "quantize": dict(
        stage=4, label="Banding",
        summary="Rounds the picture to fewer levels, so smooth gradients band.",
        help="Optional dithering hides bands under fine noise. Put it before Compression.",
        ports={"clip": VIDEO_IN}),
    # The analogue video chain: one device per node, in signal order.
    "ntsc_encode": dict(
        stage=4, chain="ntsc", label="Composite Encode",
        summary="Turns the picture into an NTSC composite signal, with any off-air snow and ghosts.",
        help="The start of an analogue chain: Encode, VHS Tape, Playback Deck, Composite Decode. Meant "
             "for LR near SD size, after Resample: a tape holds about 480 lines. A signal previews as the "
             "chain so far, played on a consumer deck if it has no deck yet, through a plain decoder.",
        ports={"clip": VIDEO_IN, "signal": SIGNAL}),
    "ntsc_tape": dict(
        stage=4, chain="ntsc", label="VHS Tape",
        summary="Records the signal to a simulated VHS tape: soft, noisy, with dropouts and wobble.",
        help="Empty settings keep the tape grade's own values. A VHS Tape fed by a VHS Tape is a dub: "
             "it adds generations on the same kind of tape, so only its Generations is used. Its preview "
             "plays the tape on a consumer deck.",
        ports={"signal": SIGNAL}),
    "ntsc_deck": dict(
        stage=4, chain="ntsc", label="Playback Deck",
        summary="Plays the tape back through a VCR or capture deck.",
        help="One deck per chain, after the last tape. Without one, the tape plays on a consumer deck. "
             "One tape can feed several decks, each its own chain. Its preview uses a plain decoder.",
        ports={"signal": SIGNAL}),
    "ntsc_decode": dict(
        stage=4, chain="ntsc", label="Composite Decode",
        summary="Decodes the signal back to a picture, as a TV or capture card does.",
        help="The end of an analogue chain: the whole chain is simulated here, once. Put Compression "
             "after it for a capture to disc or file.",
        ports={"signal": SIGNAL, "clip": ("Video", "The picture as captured.")}),
    "codec": dict(
        stage=4, label="Compression",
        summary="Encodes the chunk like a real release (DVD, broadcast, web, Xvid) and decodes it.",
        help="Pick a profile; quality moves within its realistic range. DVD uses real MPEG-2 settings: "
             "buffer, GOP, B-frames, bitrate per pixel. For a rip of a rip, chain a second Compression.",
        ports={"clip": VIDEO_IN}),
    "player_deblock": dict(
        stage=4, label="Player Deblock",
        summary="Smooths the block edges as a player does, leaving soft, waxy flat areas.",
        help="Put it right after Compression. A player that also reduces noise is this, then Player DNR.",
        ports={"clip": VIDEO_IN}),
    "player_dnr": dict(
        stage=4, label="Player DNR",
        summary="Reduces noise as a player or TV does, smearing fine detail and trailing on motion.",
        help="Put it after Compression, or after Player Deblock.",
        ports={"clip": VIDEO_IN}),
    "chroma_subsample": dict(
        stage=4, label="Chroma Subsampling",
        summary="Shrinks and regrows colour as a disc and player do: blocky, blurred or offset colour.",
        help="After Compression it models the player's colour upsampling; before it, the encoder's "
             "subsampling too. The interlaced error belongs with telecined material.",
        ports={"clip": VIDEO_IN}),
    "ghost_echo": dict(
        stage=4, label="Ghost Echo",
        summary="Adds a faint echo of the picture to the right, like a reflected analogue signal.",
        help="For broadcast and tape looks. Keep it rare and faint.",
        ports={"clip": VIDEO_IN}),
    "scale_to_model": dict(
        stage=4, label="Scale to Model Size",
        summary="Scales LR to the model's input size: HR size divided by the model's scale.",
        help="The last scaling of the LR chain. Connect the HR chunk to 'HR' so the size follows it.",
        ports={"clip": VIDEO_IN, "hr": ("HR", "The HR picture; only its size is used.")}),
    "ffmpeg_filter": dict(
        stage=4, label="Custom ffmpeg Filter",
        summary="Runs any ffmpeg video filter you type, for damage no other node makes.",
        help="Takes an ffmpeg -vf filter string such as 'eq=contrast=0.9,unsharp'. The picture goes "
             "through ffmpeg losslessly, so only the filter changes it.",
        ports={"clip": VIDEO_IN}),

    # 5 Clean up
    "denoise_hr": dict(
        stage=5, label="Denoise HR (AI)",
        summary="Removes grain and noise from the HR target with a denoising model.",
        help="A clean HR target teaches the model to output clean video. Needs a 1x denoise model in ONNX "
             "format.",
        ports={"clip": ("HR", "The HR picture to clean.")}),
    "prism_deh264": dict(
        stage=5, label="Remove H.264 Artifacts (AI)",
        summary="Removes compression blocking and smearing from the HR target with a Prism model.",
        help="For HR sources that are themselves compressed, such as web releases. Needs a Prism de-H264 "
             "model in ONNX format; the strength should roughly match the source's CRF.",
        ports={"clip": ("HR", "The HR picture to clean.")}),
    "fast_line_darken": dict(
        stage=5, also=[2], label="Darken Line Art",
        summary="Darkens thin lines, for anime targets with washed-out line art.",
        help="Makes the model produce stronger lines. Use sparingly; too much makes lines look drawn on.",
        ports={"clip": ("HR", "The HR picture.")}),
    "fine_dehalo": dict(
        stage=5, also=[2], label="Remove Halos",
        summary="Removes bright and dark halos around edges from the HR target.",
        help="When the HR source has sharpening halos, the model would learn to add them. This takes them "
             "out of the target.",
        ports={"clip": ("HR", "The HR picture.")}),
    "filter_template": dict(
        stage=5, also=[2, 4], label="Filter Template",
        summary="Runs a VapourSynth filter from Vapourkit's template list: denoise, deinterlace, colour fix "
                "and more.",
        help="Pick a template and its code appears, ready to edit; leave it unedited to run the template as "
             "written. The code reads `clip` and assigns the result to it. Connect Reference for templates "
             "that compare against another picture, such as the colour fixes. Some templates change the "
             "frame rate (deinterlacers, IVTC); keep those before Split.",
        ports={"clip": VIDEO_IN, "reference": ("Reference", "The picture `original_clip` names, for colour "
                                                             "fixes. Without it, the input itself.")}),
    "tone_adjust": dict(
        stage=5, label="Brightness / Contrast / Gamma",
        summary="Fixed brightness, contrast and gamma changes.",
        help="For a whole release that is too dark, flat or bright. For matching HR to LR automatically, "
             "use Match Gamma instead.",
        ports={"clip": VIDEO_IN}),
    "match_gamma": dict(
        stage=5, label="Match Gamma",
        summary="Adjusts HR's gamma to match LR's (or the other way round), measured automatically.",
        help="Two releases are often graded differently. If HR and LR differ in brightness, the model "
             "learns to change brightness. Run this on a dataset of pairs to remove the difference.",
        ports={"hr": HR, "lr": ("LR centre frames", "The LR frame matching each HR target."),
               "lr_all": ("All LR frames", "Every LR frame, adjusted when matching LR to HR.")}),
    "anicc": dict(
        stage=5, label="Match Colours to LR (AI)",
        summary="Corrects HR's colours to match LR with the AniCC model.",
        help="Like Match Gamma, but for colour. Needs the AniCC model in ONNX format.",
        ports={"hr": HR, "lr": LR}),
    "onnx_model": dict(
        stage=5, label="AI Model (ONNX)",
        summary="Runs any single-input ONNX image model on the picture.",
        help="For any other model: denoisers, deblockers, or a model that degrades on purpose. It runs on "
             "the app's inference backend (TensorRT, DirectML, NCNN or MIGraphX).",
        ports={"clip": VIDEO_IN}),
    "static_stack": dict(
        stage=5, label="Average Still Shots (HR)",
        summary="Aligns and averages the frames of each still shot into one clean HR target.",
        help="The averaged frame keeps the picture but loses the grain. The reference is usually the "
             "denoised centre frame, which the frames are aligned to.",
        ports={"clip": VIDEO_IN, "runs": ("Still shots", "From Find Still Shots."),
               "reference": ("Reference", "The frame each shot is aligned to, usually Denoise HR's output.")}),

    # Combine: branches and masks, mostly used while degrading
    "random_branch": dict(
        stage=4, also=[5], label="Random Branch",
        summary="Picks one of its inputs per chunk, by weight. Only the picked branch runs.",
        help="Joins whole damage chains, such as a web chain and a DVD chain, so each chunk gets one "
             "coherent kind of damage instead of a mix. The weights say how often each input is picked. "
             "The pick is in the manifest and the preview.",
        ports={**{f"in_{n}": (f"Input {n}", "A branch to pick from." if n == 1 else "Optional. Another branch.")
                  for n in range(1, 5)},
               "clip": ("Picked", "The input picked for this chunk.")}),
    "blend": dict(
        stage=4, also=[5], label="Blend",
        summary="Mixes two inputs, with the mix drawn per chunk.",
        help="Gives every strength between two versions, e.g. a filtered and an unfiltered copy for partial "
             "denoising or sharpening. Blend filters, not codecs: mixed blocking is not real.",
        ports={"a": ("A", "The first version."), "b": ("B", "The second version, the same size as A."),
               "clip": ("Mix", "A and B mixed.")}),
    "masked_merge": dict(
        stage=4, also=[5], label="Masked Merge",
        summary="Shows the overlay where the mask is on, the base elsewhere.",
        help="Puts damage only where it really happens: banding in flat areas, noise in shadows, smear on "
             "motion. Make the overlay from the base, and the mask with a mask node.",
        ports={"base": ("Base", "Shown where the mask is off."),
               "overlay": ("Overlay", "Shown where the mask is on; the same size as the base."),
               "mask": ("Mask", "Where the overlay shows."), "clip": ("Merged", "Base and overlay merged.")}),
    "luma_mask": dict(
        stage=4, label="Brightness Mask",
        summary="On where the picture is between two brightness levels, such as the shadows.",
        help="Use it to put noise or crushed detail in the shadows, or clipping in highlights.",
        ports={"clip": ("Video", "The picture to measure."), "mask": ("Mask", "On in the chosen brightness range.")}),
    "edge_mask": dict(
        stage=4, label="Edge Mask",
        summary="On along strong edges and lines.",
        help="Use it for damage that sits on edges, such as halos, ringing or mosquito noise.",
        ports={"clip": ("Video", "The picture to measure."), "mask": ("Mask", "On along edges.")}),
    "flat_mask": dict(
        stage=4, label="Flat Area Mask",
        summary="On in smooth areas with little detail, such as skies and gradients.",
        help="Use it for banding and blocking, which show in flat areas first.",
        ports={"clip": ("Video", "The picture to measure."), "mask": ("Mask", "On in flat areas.")}),
    "motion_mask": dict(
        stage=4, label="Motion Mask",
        summary="On where the picture changes from the previous frame.",
        help="Use it for damage only moving areas get: noise-reduction smear, ghosting, combing.",
        ports={"clip": ("Video", "The picture to measure."), "mask": ("Mask", "On where there is motion.")}),
    "mask_math": dict(
        stage=4, label="Combine Masks",
        summary="Inverts a mask, or joins two: both, either, or A but not B.",
        help="For example, flat areas that are also dark: Flat Area Mask and Brightness Mask with 'both'.",
        ports={"a": ("Mask A", "The first mask."), "b": ("Mask B", "Optional. The second mask."),
               "mask": ("Mask", "The result.")}),

    # 6 Pick
    "make_sequences": dict(
        stage=6, label="Cut into Sequences",
        summary="Cuts each chunk into runs of frames, the training samples.",
        help="Connect the clean video to HR and the degraded video to LR. A sequence is a run of frames "
             "(5 by default); 'centre' writes only the middle HR frame, as video models with one target "
             "frame expect.",
        ports={"hr": HR, "lr": LR}, outputs={"sequences": SEQUENCES}),
    "pair_sequences": dict(
        stage=6, label="Pick Aligned Sequences",
        summary="Picks the run with the most motion in each chunk of an aligned pair: LR frames, HR centre.",
        help="The Pick step for pairs made from two releases. Follow it with Cut into Tiles set to "
             "'detail', which crops where detail and motion are, then Drop Misaligned.",
        ports={"hr": ("HR", "The aligned HR chunk."), "lr": ("LR", "The synced LR chunk.")},
        outputs={"sequences": SEQUENCES}),
    "static_centers": dict(
        stage=3, label="Still Shot Centre Frames (LR)",
        summary="The untouched middle frame of each still shot, used as the noisy LR image.",
        help="Pairs with Average Still Shots: the centre frame is the noisy input, the average the clean "
             "target.",
        ports={"clip": VIDEO_IN, "runs": ("Still shots", "From Find Still Shots.")}),

    # 7 Filter
    "reject_blank": dict(
        stage=7, label="Drop Blank Sequences",
        summary="Drops sequences with black, flat or nearly detail-free frames.",
        help="Fades, black frames and flat skies teach the model nothing. Put filters after the Pick step "
             "and before Save.",
        ports={"sequences": SEQUENCES}),
    "reject_motion": dict(
        stage=7, label="Drop Too Still / Too Fast",
        summary="Drops sequences with almost no motion or with too much (cuts, flashes).",
        help="Video models learn from motion between frames; still sequences waste training time and "
             "sequences across a cut or flash confuse the model.",
        ports={"sequences": SEQUENCES}),
    "reject_incomplete": dict(
        stage=7, label="Drop Incomplete Sequences",
        summary="Drops sequences that are missing frames.",
        help="A safety check before Save, so every sequence has exactly the expected number of frames.",
        ports={"sequences": SEQUENCES}),
    "reject_misaligned": dict(
        stage=7, label="Drop Misaligned",
        summary="Drops pair sequences that drifted out of sync or whose HR and LR do not line up.",
        help="For pairs made from two releases, after Cut into Tiles. A failed warp teaches the model to "
             "blur. Ask Cut into Tiles for a few spare crops; Crops per run keeps the best ones that pass.",
        ports={"sequences": SEQUENCES}),
    "dedup_sequences": dict(
        stage=7, label="Drop Near-Duplicates",
        summary="Drops sequences that look almost the same as one already kept.",
        help="TV shows repeat shots, openings and long holds. Removing duplicates keeps the dataset varied. "
             "Needs the ConvNeXt weights file.",
        ports={"sequences": SEQUENCES}),
    "tile": dict(
        stage=7, label="Cut into Tiles",
        summary="Cuts every sequence into matching HR and LR tiles: on a grid, or where detail is.",
        help="Smaller samples train faster and fit in memory. LR tiles are sized from the HR tile by the "
             "scale between the two. 'detail' takes the most detailed tiles, drawn to where the picture "
             "moves, as pair mode does.",
        ports={"sequences": SEQUENCES}),

    # 8 Save
    "write_dataset": dict(
        stage=8, label="Save Dataset",
        summary="Writes every sequence as HR and LR images, plus a manifest of the settings used.",
        help="The last node of a job. Files are named showN_FrameM in HR and LR folders, the layout "
             "traiNNer-redux expects. A rerun adds to the folder instead of overwriting it. Set a 'bad' "
             "folder to also keep rejected sequences for review. Variants write each sequence several times, each "
             "with fresh LR damage; a subfolder lets several writers share one dataset folder.",
        ports={"sequences": SEQUENCES}),
    "write_processed": dict(
        stage=8, label="Save Cleaned-up Dataset",
        summary="Writes the cleaned-up HR targets to hr_p/ (and adjusted LR to lr_p/) beside the originals.",
        help="The last node of a clean-up job started from Load Existing Dataset. The original hr/ and lr/ "
             "are never changed.",
        ports={"hr": ("HR", "The cleaned-up HR targets."), "lr_all": ("All LR frames", "Optional adjusted LR.")}),
    "write_static_dataset": dict(
        stage=8, label="Save Denoise Dataset",
        summary="Writes each still shot's averaged frame (HR) and centre frame (LR).",
        help="The last node of a denoise dataset job.",
        ports={"hr": ("HR", "From Average Still Shots."), "lr": ("LR", "From Still Shot Centre Frames."),
               "runs": ("Still shots", "From Find Still Shots.")}),

    # Values: no stage, they feed settings
    "random_number": dict(
        stage=None, group="Values", label="Random Number",
        summary="A random number per chunk, between low and high.",
        help="Wire it into several settings to give them the same draw, or through Math to derive one "
             "setting from another. It is drawn again for every chunk, the same way on every rerun.",
        outputs={"value": VALUE}),
    "random_pick": dict(
        stage=None, group="Values", label="Random Pick",
        summary="Picks one of a list of options per chunk, optionally weighted.",
        help="Wire it into a dropdown, then tick the options to pick from. Wired into two nodes, both get "
             "the same pick on every chunk, such as two Compression nodes that share a profile.",
        outputs={"value": VALUE}),
    "math": dict(
        stage=None, group="Values", label="Math",
        summary="Adds, subtracts, multiplies or divides two numbers, or takes the smaller or larger.",
        help="Link a value into A or B, or type a number. For example, a re-encode quality that is the "
             "first encode's quality times a random factor below 1. Min and max keep a result in range.",
        outputs={"value": VALUE}),
    "if_equals": dict(
        stage=None, group="Values", label="If Equals",
        summary="Gives one value when its input equals another, and a second value otherwise.",
        help="For settings that depend on each other: if the profile is dvd_mpeg2, the size is 720x480. "
             "Link the tested value into Value, and wire the output into the setting it decides.",
        outputs={"value": VALUE}),
}


def docs_for(type_name):
    return NODE_DOCS.get(type_name)
