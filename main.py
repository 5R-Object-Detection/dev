import cv2
import time
import numpy as np
from gui import launch_gui
from analyzer import BehavioralAnalyzer
from validators import PhoneToPersonAssociator, WristDistanceValidator, CentroidMotionTracker, PocketHandsValidator
from fence import VirtualFence

if __name__ == "__main__":
    config = launch_gui()
    video_source = config.get("source")
    if video_source is None:
        print("Pengoperasian dibatalkan.")
        exit()

    associator = PhoneToPersonAssociator()
    # Assuming wrist threshold of 100px (increased for better webcam tolerance)
    validator = WristDistanceValidator(threshold_pixels=100.0)
    # Assumes a displacement of >10px over 15 frames is walking (decreased for indoor walking)
    tracker = CentroidMotionTracker(buffer_size=15, displacement_threshold=10.0)
    pocket_validator = PocketHandsValidator(buffer_size=15)
    
    analyzer = BehavioralAnalyzer(
        pose_model_path="yolo11n-pose.pt", 
        detect_model_path="yolo11n.pt", 
        associator=associator,
        validator=validator,
        tracker=tracker,
        pocket_v=pocket_validator,
        fences=[]  # Mulai dengan virtual fence kosong
    )
    
    run_type = config.get("type", "video")
    enable_fence = config.get("enable_fence", True)
    enable_phone = config.get("enable_phone", True)
    enable_pocket = config.get("enable_pocket", True)
    
    window_name = "Toyota 5R: Behavioral Dashboard"
    cv2.namedWindow(window_name)
    
    # State Engine untuk Dynamic Region Selector
    drawing_mode = False
    frozen_frame = None
    fence_points = []
    
    def mouse_callback(event, x, y, flags, param):
        global drawing_mode, fence_points, frozen_frame
        if drawing_mode and event == cv2.EVENT_LBUTTONDOWN:
            fence_points.append((x, y))

    cv2.setMouseCallback(window_name, mouse_callback)
    
    if run_type == "image":
        print(f"Loading image: {video_source}")
        original_img = cv2.imread(video_source)
        if original_img is None:
            print(f"Failed to load image: {video_source}")
            exit()
            
        print("Membuka mode gambar. Tekan 'S' untuk menyimpan, 'R' untuk menggambar area, dan 'Q' untuk keluar.")
        
        while True:
            display_frame = original_img.copy()
            
            if drawing_mode:
                for pt in fence_points:
                    cv2.circle(display_frame, pt, 5, (0, 255, 0), -1)
                if len(fence_points) > 1:
                    cv2.line(display_frame, fence_points[-2], fence_points[-1], (0, 255, 0), 2)
                    cv2.polylines(display_frame, [np.array(fence_points, np.int32)], False, (0, 200, 0), 2)
                
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], 100), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
                
                cv2.putText(display_frame, "MODE GAMBAR VIRTUAL FENCE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                cv2.putText(display_frame, "Klik titik untuk menggambar. Tekan ENTER untuk selesai / ESC batal.", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (13, 32): # Enter
                    if len(fence_points) >= 3:
                        analyzer.fences = [VirtualFence(polygon_points=fence_points)]
                    drawing_mode = False
                elif key == 27: # Esc
                    drawing_mode = False
                continue
                
            # Mode Tampil: Proses inferensi dengan is_image=True
            alert_ids, fence_violator_ids, pocket_violator_ids, annotated_frame, debug_distances = analyzer.process_frame(
                original_img, is_image=True, 
                enable_fence=enable_fence, enable_phone=enable_phone, enable_pocket=enable_pocket
            )
            
            is_alerting = bool(alert_ids) or bool(fence_violator_ids) or bool(pocket_violator_ids)
            
            # --- UI ENHANCEMENTS UNTUK ALERT MASSIVE ---
            if is_alerting:
                cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1]-1, annotated_frame.shape[0]-1), (0, 0, 255), 20)
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (0, 0), (annotated_frame.shape[1], 120), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
                
                text_size = cv2.getTextSize("!!! PELANGGARAN TERDETEKSI !!!", cv2.FONT_HERSHEY_DUPLEX, 1.3, 3)[0]
                text_x = (annotated_frame.shape[1] - text_size[0]) // 2
                cv2.putText(annotated_frame, "!!! PELANGGARAN TERDETEKSI !!!", (text_x, 70), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 3)

            y_offset = 160 if is_alerting else 80
            if alert_ids:
                cv2.putText(annotated_frame, f">> PENGGUNAAN HP (ID: {alert_ids})", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 3)
                y_offset += 40
            if fence_violator_ids:
                cv2.putText(annotated_frame, f">> PELANGGARAN FENCE (ID: {fence_violator_ids})", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                y_offset += 40
            if pocket_violator_ids:
                cv2.putText(annotated_frame, f">> TANGAN DI SAKU CELANA (ID: {pocket_violator_ids})", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)
                y_offset += 40
                
            for p_id, dist in debug_distances.items():
                if dist != float('inf'):
                    cv2.putText(annotated_frame, f"Deteksi ID {p_id} (Jarak Tangan-HP: {dist:.1f}px)", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                    y_offset += 30
            
            # Footer
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, annotated_frame.shape[0]-60), (annotated_frame.shape[1], annotated_frame.shape[0]), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
            cv2.putText(annotated_frame, "R: Gambar Fence | S: Simpan Gambar | Q: Keluar", (20, annotated_frame.shape[0]-25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(window_name, annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key in (27, ord('q'), ord('Q')):
                break
            elif key in (ord('s'), ord('S')):
                save_path = f"inference_result_{int(time.time())}.jpg"
                clean_save_frame = annotated_frame.copy()
                # Remove overlay footer from saved image text if desired, or keep it.
                cv2.imwrite(save_path, clean_save_frame)
                print(f"Gambar berhasil disimpan ke: {save_path}")
                
            elif key in (ord('r'), ord('R')):
                drawing_mode = True
                fence_points = []
                
        cv2.destroyAllWindows()
        exit()

    else:
        # ======= VIDEO MODE =======
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Failed to load video source: {video_source}")
            exit()
            
        prev_t = time.time()
        
        # State for Alert Holding
        ALERT_DISPLAY_DURATION = 3.0 # Tampilkan alert selama 3 detik
        last_phone_alert_time = 0.0
        last_fence_alert_time = 0.0
        last_pocket_alert_time = 0.0
        last_alert_ids = []
        last_fence_violator_ids = []
        last_pocket_violator_ids = []
            
        while True:
            if drawing_mode:
                # Mode pause dan gambar: tampilkan frame yang membeku
                display_frame = frozen_frame.copy()
                for pt in fence_points:
                    cv2.circle(display_frame, pt, 5, (0, 255, 0), -1)
                if len(fence_points) > 1:
                    cv2.line(display_frame, fence_points[-2], fence_points[-1], (0, 255, 0), 2)
                    cv2.polylines(display_frame, [np.array(fence_points, np.int32)], False, (0, 200, 0), 2)
                    
                # Layar Bantuan Teks Di Atas
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], 100), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
                    
                cv2.putText(display_frame, "MODE GAMBAR VIRTUAL FENCE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                cv2.putText(display_frame, "Klik titik untuk menggambar. Tekan ENTER untuk selesai / ESC batal.", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key in (13, 32): # Enter or Space
                    if len(fence_points) >= 3:
                        new_fence = VirtualFence(polygon_points=fence_points)
                        analyzer.fences = [new_fence]
                    drawing_mode = False
                elif key == 27: # Esc
                    drawing_mode = False
                continue

            ret, frame = cap.read()
            if not ret:
                break
                
            alert_ids, fence_violator_ids, pocket_violator_ids, annotated_frame, debug_distances = analyzer.process_frame(
                frame, is_image=False,
                enable_fence=enable_fence, enable_phone=enable_phone, enable_pocket=enable_pocket
            )
            
            curr_t = time.time()
            fps = 1 / (curr_t - prev_t) if curr_t > prev_t else 0
            prev_t = curr_t
            
            # --- ALERT PERSISTENCE LOGIC ---
            if alert_ids:
                last_phone_alert_time = curr_t
                last_alert_ids = alert_ids
                
            if fence_violator_ids:
                last_fence_alert_time = curr_t
                last_fence_violator_ids = fence_violator_ids
                
            if pocket_violator_ids:
                last_pocket_alert_time = curr_t
                last_pocket_violator_ids = pocket_violator_ids
                
            is_phone_alerting = (curr_t - last_phone_alert_time) < ALERT_DISPLAY_DURATION
            is_fence_alerting = (curr_t - last_fence_alert_time) < ALERT_DISPLAY_DURATION
            is_pocket_alerting = (curr_t - last_pocket_alert_time) < ALERT_DISPLAY_DURATION
            is_alerting_visually = is_phone_alerting or is_fence_alerting or is_pocket_alerting
            
            # --- UI ENHANCEMENTS UNTUK ALERT MASSIVE ---
            if is_alerting_visually:
                # Buat pinggiran layar menjadi MERAH berkedip secara masif
                cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1]-1, annotated_frame.shape[0]-1), (0, 0, 255), 20)
                
                # Buat Banner Merah Lebar Di Atas
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (0, 0), (annotated_frame.shape[1], 120), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
                
                # Teks Peringatan Super Besar
                text_size = cv2.getTextSize("!!! PELANGGARAN TERDETEKSI !!!", cv2.FONT_HERSHEY_DUPLEX, 1.3, 3)[0]
                text_x = (annotated_frame.shape[1] - text_size[0]) // 2
                cv2.putText(annotated_frame, "!!! PELANGGARAN TERDETEKSI !!!", (text_x, 70), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 3)

            # Print detailed violations below banner or top corner
            y_offset = 160 if is_alerting_visually else 80
            if is_phone_alerting and last_alert_ids:
                cv2.putText(annotated_frame, f">> PENGGUNAAN HP SAAT BERJALAN (ID: {last_alert_ids})", 
                            (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 3)
                y_offset += 40
                            
            if is_fence_alerting and last_fence_violator_ids:
                cv2.putText(annotated_frame, f">> PELANGGARAN VIRTUAL FENCE (ID: {last_fence_violator_ids})", 
                            (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                y_offset += 40
                
            if is_pocket_alerting and last_pocket_violator_ids:
                cv2.putText(annotated_frame, f">> TANGAN DI SAKU CELANA TERDETEKSI (ID: {last_pocket_violator_ids})", 
                            (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)
                y_offset += 40
                
            # Display debug info
            for p_id, dist in debug_distances.items():
                if dist != float('inf'):
                    cv2.putText(annotated_frame, f"Deteksi ID {p_id} (Jarak Tangan-HP: {dist:.1f}px)", 
                                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                    y_offset += 30
                            
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (annotated_frame.shape[1]-150, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # --- UI FOOTER UNTUK KONTROL ---
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, annotated_frame.shape[0]-60), (annotated_frame.shape[1], annotated_frame.shape[0]), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
            cv2.putText(annotated_frame, "Tekan 'R' tuk menggambar ulang virtual fence | Tekan 'Q' tuk keluar", (20, annotated_frame.shape[0]-25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(window_name, annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key in (27, ord('q'), ord('Q')):
                break
            elif key in (ord('r'), ord('R')):
                # Trigger state ganti Virtual Fence
                drawing_mode = True
                frozen_frame = frame.copy()
                fence_points = []
                
        cap.release()
        cv2.destroyAllWindows()
