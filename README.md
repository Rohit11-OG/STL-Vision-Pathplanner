<div align="center">

<!-- Animated Header -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=300&section=header&text=🚀%20STL%20Vision%20PathPlanner&fontSize=50&fontColor=fff&animation=twinkling&fontAlignY=35&desc=From%20CAD%20to%20Camera%20to%20Robot%20—%20All%20in%20One!&descAlignY=55&descSize=20"/>

<!-- Typing Animation -->
<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=28&duration=3000&pause=1000&color=00F7FF&center=true&vCenter=true&multiline=true&repeat=false&random=false&width=800&height=100&lines=🤖+Transform+3D+Models+into+Robotic+Paths;🧠+AI-Powered+Real-Time+Detection" alt="Typing SVG" /></a>

<!-- Animated Badges -->
<p>
<img src="https://img.shields.io/badge/Python-3.8+-FFD43B?style=for-the-badge&logo=python&logoColor=blue&labelColor=black"/>
<img src="https://img.shields.io/badge/YOLOv8-Ultralytics-FF6F61?style=for-the-badge&logo=yolo&logoColor=white&labelColor=black"/>
<img src="https://img.shields.io/badge/OpenCV-Real--Time-00FF00?style=for-the-badge&logo=opencv&logoColor=white&labelColor=black"/>
<img src="https://img.shields.io/badge/ROS2-Robot%20Ready-FF6600?style=for-the-badge&logo=ros&logoColor=white&labelColor=black"/>
</p>

<!-- Activity Graph -->
<img src="https://github-readme-activity-graph.vercel.app/graph?username=Rohit11-OG&theme=react-dark&hide_border=true&area=true" width="90%"/>

</div>

---

<div align="center">

## ⚡ LIGHTNING FAST PIPELINE ⚡

```
    ╔══════════════════════════════════════════════════════════════╗
    ║  📦 STL  ──▶  🎨 TRAIN  ──▶  👁️ DETECT  ──▶  🛤️ PATH  ──▶  🤖 ROBOT  ║
    ╚══════════════════════════════════════════════════════════════╝
```

</div>

---

## � WHAT MAKES THIS INSANE?

<table>
<tr>
<td width="50%">

### � Zero Manual Labeling
```diff
+ Auto-generates training data from STL
+ No tedious bounding box annotation
+ 500+ synthetic images in seconds
```

</td>
<td width="50%">

### 🎯 6 Path Strategies
```diff
! Contour  → Inspection/Welding
! Spiral   → Polishing/Coating
! Zigzag   → Full Coverage
! Surface  → Depth-Aware
! Grid     → Scanning
! Approach → Pick & Place
```

</td>
</tr>
</table>

---

<div align="center">

## 🎮 KEYBOARD SHORTCUTS

<img src="https://img.shields.io/badge/P-Generate%20Path-00FF00?style=flat-square&labelColor=1a1a2e"/>
<img src="https://img.shields.io/badge/1--6-Switch%20Strategy-FF6600?style=flat-square&labelColor=1a1a2e"/>
<img src="https://img.shields.io/badge/V-Toggle%20Viz-00FFFF?style=flat-square&labelColor=1a1a2e"/>
<img src="https://img.shields.io/badge/R-Reload%20Config-FF00FF?style=flat-square&labelColor=1a1a2e"/>
<img src="https://img.shields.io/badge/S-Save%20Frame-FFFF00?style=flat-square&labelColor=1a1a2e"/>
<img src="https://img.shields.io/badge/Q-Quit-FF0000?style=flat-square&labelColor=1a1a2e"/>

</div>

---

## 🛠️ QUICK START

```bash
# 🔽 Clone the repo
git clone https://github.com/Rohit11-OG/STL-Vision-Pathplanner-.git
cd STL-Vision-Pathplanner-

# 📦 Install dependencies
pip install -r requirements.txt

# 🚀 Full pipeline: STL → Train → Detect → Path
python main.py full --stl your_object.stl --epochs 50

# 🎯 Or run path generation directly
python main.py path --strategy spiral --camera 0
```

---

## 🌀 PATH STRATEGIES

<div align="center">

| Strategy | Visual | Use Case |
|:--------:|:------:|:---------|
| **Contour** | 🔵⭕ | Inspection, Welding edges |
| **Spiral** | 🌀 | Polishing, Painting inward |
| **Zigzag** | ⚡ | Complete surface coverage |
| **Surface** | 🌊 | Following 3D depth contours |
| **Grid** | ▦ | Scanning, Uniform coating |
| **Approach** | 📍 | Pick and place operations |

</div>

---

## 📁 PROJECT STRUCTURE

```
🗂️ STL-Vision-Pathplanner/
├── 🎯 main.py                 # CLI entry point
├── 🧠 train_detector.py       # YOLO training pipeline
├── 📷 realtime_detector.py    # Live detection + visualization
├── 🛤️ tool_path_planner.py    # Path generation engine
├── 🎨 data_generator.py       # Synthetic data from STL
├── ⚙️ config.py               # System configuration
├── 📝 settings.yaml           # User-editable settings
├── 🤖 ros2_path_publisher.py  # ROS2 integration node
└── 📦 stl_detector_ros2/      # Full ROS2 package
```

---

## 🤖 ROS2 INTEGRATION

<div align="center">

```yaml
# 📡 Topics Published
/tool_path          → nav_msgs/Path
/detections         → DetectionArray

# 🔧 Services Available
/publish_latest_path → std_srvs/Trigger
```

</div>

```bash
# Launch the detection node
ros2 launch stl_detector_ros2 detection.launch.py

# Echo the path topic
ros2 topic echo /tool_path
```

---

## � OUTPUT FORMAT

```yaml
header:
  frame_id: "camera_link"
  stamp: "2026-01-29T18:00:00"

path:
  waypoints:
    - pose:
        position: {x: 0.15, y: 0.05, z: 0.55}
        orientation: {x: 0, y: 0, z: 0.38, w: 0.92}
      velocity: 0.1
      time_from_start: {sec: 0, nanosec: 500000000}
```

---

<div align="center">

## 🛠️ TECH STACK

<p>
<img src="https://skillicons.dev/icons?i=python,pytorch,opencv,ros,linux,git,vscode&theme=dark" />
</p>

</div>

---

## 🤝 CONTRIBUTING

<div align="center">

```
   🍴 Fork  →  🌿 Branch  →  💻 Code  →  📤 PR  →  🎉 Merge!
```

</div>

1. Fork the repository
2. Create feature branch: `git checkout -b feature/AwesomeFeature`
3. Commit changes: `git commit -m '✨ Add AwesomeFeature'`
4. Push: `git push origin feature/AwesomeFeature`
5. Open Pull Request

---

<div align="center">

## ⭐ STAR THIS REPO IF YOU FIND IT USEFUL!

<img src="https://img.shields.io/github/stars/Rohit11-OG/STL-Vision-Pathplanner-?style=social"/>
<img src="https://img.shields.io/github/forks/Rohit11-OG/STL-Vision-Pathplanner-?style=social"/>
<img src="https://img.shields.io/github/watchers/Rohit11-OG/STL-Vision-Pathplanner-?style=social"/>

---

### Made with ❤️ and ☕ by [Rohit](https://github.com/Rohit11-OG)

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer"/>

</div>
