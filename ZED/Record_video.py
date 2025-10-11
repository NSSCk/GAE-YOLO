import pyzed.sl as sl

def convert_svo_to_mp4(svo_file, output_file):

    zed = sl.Camera()


    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_file)  # 从 SVO 文件加载


    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Failed to open SVO file:", err)
        exit(1)


    runtime_params = sl.RuntimeParameters()
    image = sl.Mat()
    writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), zed.get_camera_information().camera_fps,
                             (zed.get_camera_information().camera_resolution.width, zed.get_camera_information().camera_resolution.height))


    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()
            writer.write(frame)
        else:
            break


    writer.release()
    zed.close()
    print("Conversion completed.")

if __name__ == "__main__":
    svo_file = "output.svo"
    output_file = "output.mp4"
    convert_svo_to_mp4(svo_file, output_file)