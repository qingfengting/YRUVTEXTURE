
// Created by xiaok on 16-12-16.
//

#include "../../../lib/local/LandmarkDetector/include/LandmarkCoreIncludes.h"

#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"
#include "IO/ObjIO.hpp"
#include "fitting/RenderingParameters.hpp"
#include "fitting/nonlinear_camera_estimation.hpp"
#include "fitting/test_function.hpp"
#include "fitting/detail/optional_cerealisation.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "render/texture_extraction.hpp"
#include "render/render_mask_image.hpp"
#include <easylogging++.h>
#include <cv.h>


namespace po = boost::program_options;
namespace fs = boost::filesystem;
using namespace yr;

//using namespace std;

using cv::Vec2f;
using cv::Vec3f;
using cv::Vec4f;

namespace opt = boost::program_options;
//OST_CORE_ENABLE_IF_HPP
//#define BOOST_NO_SFINAE


INITIALIZE_EASYLOGGINGPP

void initLogInfo(){
    el::Configurations defaultConf;
    defaultConf.setToDefault();
    defaultConf.set(el::Level::Info,el::ConfigurationType::Format,"%level %datetime %msg");
    defaultConf.set(el::Level::Info,el::ConfigurationType::ToFile,"true");
    defaultConf.set(el::Level::Info,el::ConfigurationType::Filename,"../log/fit_model/log.txt");
    defaultConf.set(el::Level::Info,el::ConfigurationType::ToStandardOutput,"true");
    el::Loggers::reconfigureLogger("default",defaultConf);
}

template<typename T, glm::precision P = glm::defaultp>
bool are_vertices_ccw_in_screen_space(const glm::tvec2<T, P>& v0, const glm::tvec2<T, P>& v1, const glm::tvec2<T, P>& v2)
{
    const auto dx01 = v1[0] - v0[0]; // todo: replace with x/y (GLM)
    const auto dy01 = v1[1] - v0[1];
    const auto dx02 = v2[0] - v0[0];
    const auto dy02 = v2[1] - v0[1];

    return (dx01*dy02 - dy01*dx02 < T(0)); // Original: (dx01*dy02 - dy01*dx02 > 0.0f). But: OpenCV has origin top-left, y goes down
};

//void draw_wireframe(cv::Mat image, const Mesh::ObjIO& mesh, glm::mat4x4 modelview, glm::mat4x4 projection, glm::vec4 viewport, cv::Scalar colour = cv::Scalar(0, 255, 0, 255))
void draw_wireframe(cv::Mat image, const Mesh::ObjIO& mesh, cv::Scalar colour = cv::Scalar(0, 255, 0, 255))
{
    for (const auto& triangle : mesh.tvi)
    {
        int vex1 = triangle[0] - 1;
        int vex2 = triangle[1] - 1;
        int vex3 = triangle[2] - 1;

        cv::line(image, cv::Point(mesh.vertices[vex1].x, image.rows - mesh.vertices[vex1].y), cv::Point(mesh.vertices[vex2].x, image.rows -mesh.vertices[vex2].y), colour);
        cv::line(image, cv::Point(mesh.vertices[vex2].x, image.rows -mesh.vertices[vex2].y), cv::Point(mesh.vertices[vex3].x, image.rows -mesh.vertices[vex3].y), colour);
        cv::line(image, cv::Point(mesh.vertices[vex3].x, image.rows -mesh.vertices[vex3].y), cv::Point(mesh.vertices[vex1].x, image.rows -mesh.vertices[vex1].y), colour);
    }
};


float diff_calc(float * vector0, float *vector1){
    float x0, y0, x1, y1, z1;
    float x00, y00, x11, y11, z11, out;
    float  t0, t1;
    x0 = vector0[0];   y0 = vector0[1];  x1 = vector0[2];   y1 = vector0[3]; z1 = vector0[4];
    x00 = vector1[0];   y00 = vector1[1];  x11 = vector1[2];   y11 = vector1[3]; z11 = vector1[4];

    t0 = (x0 - x00)*(x0 - x00) + (y0 - y00)*(y0 - y00);
    t1 = (x1 - x11)*(x1 - x11) + (y1 - y11)*(y1 - y11) ;//+ (z1 - z11)*(z1 - z11);

    //if(isinf(t0) || isnan(t0) || isnan(-t0)) cout<<"diff_calc Input Error!"<<endl;
    //if(isinf(t1) || isnan(t0)) cout<<"diff_calc Input Error!"<<endl;
    //if(t1 == 0) cout<<"diff_calc Input Error!"<<endl;
   out = sqrt(t0 / t1);
}
/*
void register_diff( ObjIO model,Mark::MapperCollection vec_map, cv::Mat out_put) {//string path_uv_mapper
    //Mark::MarkMapper mapper;
    //Mark::MapperCollection vec_map;
    //Mark::ImportFromFile(vec_map,path_uv_mapper);

    //ofstream outfile;
    boost::filesystem::ofstream outfile;
    outfile.open("register_file.txt", ios::out);

    float feature_pt_1_data[5], feature_pt_2_data[5];
    string feature_pt[23][2];//feature_pt_2;
    float re_1[23][2];//0~8: contour,9~12:eye, 13~16: nose 17~20:mounth

    feature_pt[0][0] = "contour_left1"; feature_pt[0][1] = "contour_right1";     feature_pt[1][0] = "contour_left2"; feature_pt[1][1] = "contour_right2";  feature_pt[2][0] = "contour_left3"; feature_pt[2][1] = "contour_right3";
    feature_pt[3][0] = "contour_left4"; feature_pt[3][1] = "contour_right4";     feature_pt[4][0] = "contour_left5"; feature_pt[4][1] = "contour_right5";  feature_pt[5][0] = "contour_left6"; feature_pt[5][1] = "contour_right6";
    feature_pt[6][0] = "contour_left7"; feature_pt[6][1] = "contour_right7";     feature_pt[7][0] = "contour_left8"; feature_pt[7][1] = "contour_right8";  feature_pt[8][0] = "contour_left9"; feature_pt[8][1] = "contour_right9";
    feature_pt[9][0] = "left_eye_left_corner"; feature_pt[9][1] = "left_eye_right_corner";     feature_pt[10][0] = "left_eye_bottom"; feature_pt[10][1] = "left_eye_top";  feature_pt[11][0] = "right_eye_left_corner"; feature_pt[11][1] = "right_eye_right_corner";
    feature_pt[12][0] = "right_eye_bottom"; feature_pt[12][1] = "right_eye_top";     feature_pt[13][0] = "nose_left"; feature_pt[13][1] = "nose_right";  feature_pt[14][0] = "nose_contour_left2"; feature_pt[14][1] = "nose_contour_right2";
    feature_pt[15][0] = "nose_contour_left1"; feature_pt[15][1] = "nose_contour_left3";     feature_pt[16][0] = "nose_contour_right1"; feature_pt[16][1] = "nose_contour_right3";feature_pt[17][0] = "mouth_upper_lip_left_contour2"; feature_pt[17][1] = "mouth_upper_lip_right_contour2";
    feature_pt[18][0] = "mouth_left_corner"; feature_pt[18][1] = "mouth_right_corner";     feature_pt[19][0] = "mouth_lower_lip_left_contour2"; feature_pt[19][1] = "mouth_lower_lip_right_contour2";  feature_pt[20][0] = "mouth_upper_lip_left_contour2"; feature_pt[20][1] = "mouth_lower_lip_left_contour2";
    feature_pt[21][0] = "mouth_upper_lip_top"; feature_pt[21][1] = "mouth_lower_lip_bottom";feature_pt[22][0] = "mouth_upper_lip_right_contour2"; feature_pt[22][1] = "mouth_lower_lip_right_contour2";
    for(int i=0; i<23; i++){
        for(auto m : vec_map){
            if(m.feature_name == feature_pt[i][0]){
                feature_pt_1_data[0] = m.pixel_x;
                feature_pt_1_data[1] = m.pixel_y;
                feature_pt_1_data[2] = model.vertices[m.model_point_id - 1].x;
                feature_pt_1_data[3] = model.vertices[m.model_point_id - 1].y;
                feature_pt_1_data[4] = model.vertices[m.model_point_id - 1].z;
            }
            if(m.feature_name == feature_pt[i][1]){
                feature_pt_2_data[0] = m.pixel_x;
                feature_pt_2_data[1] = m.pixel_y;
                feature_pt_2_data[2] = model.vertices[m.model_point_id - 1].x;
                feature_pt_2_data[3] = model.vertices[m.model_point_id - 1].y;
                feature_pt_2_data[4] = model.vertices[m.model_point_id - 1].z;
            }
//            cout<<"m.feature_name  "<< m.feature_name <<endl;
//           cout<<"m.feature_id = "<<m.feature_id<<endl;
        }//auto m
//        printf("i = %d\n", i);
//        printf("feature_pt_1_data[0]=%f, feature_pt_1_data[1]=%f, feature_pt_1_data[2]=%f, feature_pt_1_data[3]=%f, feature_pt_1_data[4]=%f\n",feature_pt_1_data[0], feature_pt_1_data[1], feature_pt_1_data[2], feature_pt_1_data[3], feature_pt_1_data[4]);
//        printf("feature_pt_2_data[0]=%f, feature_pt_2_data[1]=%f, feature_pt_2_data[2]=%f, feature_pt_2_data[3]=%f, feature_pt_2_data[4]=%f\n",feature_pt_2_data[0], feature_pt_2_data[1], feature_pt_2_data[2], feature_pt_2_data[3], feature_pt_2_data[4]);

        if(i <= 8){
            re_1[i][0] = 0;
            re_1[i][1] =  diff_calc(feature_pt_1_data, feature_pt_2_data);
        } else if(i <= 12){
            re_1[i][0] = 1;
            re_1[i][1] =  diff_calc(feature_pt_1_data, feature_pt_2_data);
        } else if(i <= 16){
            re_1[i][0] = 2;
            re_1[i][1] =  diff_calc(feature_pt_1_data, feature_pt_2_data);
        } else{
            re_1[i][0] = 3;
            re_1[i][1] =  diff_calc(feature_pt_1_data, feature_pt_2_data);
        }

//        cout<<i<<" re_1 = "<< re_1[i][0]<<" value "<<re_1[i][1] <<endl;
        if(outfile.is_open())
            outfile<<re_1[i][0]<<" "<<re_1[i][1]<<endl;
    }// i
    outfile.close();
}
*/



void register_diff( ObjIO model,Mark::MapperCollection vec_map, float arr[][2]) {//string path_uv_mapper string outputfile
    //Mark::MarkMapper mapper;
    //Mark::MapperCollection vec_map;
    //Mark::ImportFromFile(vec_map,path_uv_mapper);
    int dbg_flag = 0;
    //ofstream outfile;
    //outfile.open(outputfile, ios::out);

    float feature_pt_1_data[5], feature_pt_2_data[5];
    string feature_pt[23][2];//feature_pt_2;
    float re_1[23][2];//0~8: contour,9~12:eye, 13~16: nose 17~20:mounth

    feature_pt[0][0] = "contour_left1"; feature_pt[0][1] = "contour_right1";     feature_pt[1][0] = "contour_left2"; feature_pt[1][1] = "contour_right2";  feature_pt[2][0] = "contour_left3"; feature_pt[2][1] = "contour_right3";
    feature_pt[3][0] = "contour_left4"; feature_pt[3][1] = "contour_right4";     feature_pt[4][0] = "contour_left5"; feature_pt[4][1] = "contour_right5";  feature_pt[5][0] = "contour_left6"; feature_pt[5][1] = "contour_right6";
    feature_pt[6][0] = "contour_left7"; feature_pt[6][1] = "contour_right7";     feature_pt[7][0] = "left_eyebrow_lower_left_middle"; feature_pt[7][1] = "left_eye_bottom";  feature_pt[8][0] = "left_eye_top"; feature_pt[8][1] = "left_eye_bottom";
    feature_pt[9][0] = "left_eye_right_corner"; feature_pt[9][1] = "nose_contour_left1";     feature_pt[10][0] = "right_eyebrow_lower_middle"; feature_pt[10][1] = "right_eye_bottom";  feature_pt[11][0] = "right_eye_top"; feature_pt[11][1] = "right_eye_bottom";
    feature_pt[12][0] = "right_eye_left_corner"; feature_pt[12][1] = "nose_contour_right1";     feature_pt[13][0] = "nose_left"; feature_pt[13][1] = "nose_right";  feature_pt[14][0] = "nose_contour_left2"; feature_pt[14][1] = "nose_contour_right2";
    feature_pt[15][0] = "nose_contour_left1"; feature_pt[15][1] = "nose_contour_left3";     feature_pt[16][0] = "nose_contour_right1"; feature_pt[16][1] = "nose_contour_right3";feature_pt[17][0] = "mouth_upper_lip_left_contour2"; feature_pt[17][1] = "mouth_upper_lip_right_contour2";
    feature_pt[18][0] = "mouth_left_corner"; feature_pt[18][1] = "mouth_right_corner";     feature_pt[19][0] = "mouth_lower_lip_left_contour2"; feature_pt[19][1] = "mouth_lower_lip_right_contour2";  feature_pt[20][0] = "nose_contour_left1"; feature_pt[20][1] = "mouth_upper_lip_left_contour3";
    feature_pt[21][0] = "nose_contour_right1"; feature_pt[21][1] = "mouth_upper_lip_right_contour3";feature_pt[22][0] = "mouth_upper_lip_right_contour2"; feature_pt[22][1] = "mouth_lower_lip_right_contour2";



    for(int i=0; i<23; i++){
        for(auto m : vec_map){
            if(m.feature_name == feature_pt[i][0]){
                feature_pt_1_data[0] = m.pixel_x;
                feature_pt_1_data[1] = m.pixel_y;
                feature_pt_1_data[2] = model.vertices[m.model_point_id - 1].x;
                feature_pt_1_data[3] = model.vertices[m.model_point_id - 1].y;
                feature_pt_1_data[4] = model.vertices[m.model_point_id - 1].z;
            }
            if(m.feature_name == feature_pt[i][1]){
                feature_pt_2_data[0] = m.pixel_x;
                feature_pt_2_data[1] = m.pixel_y;
                feature_pt_2_data[2] = model.vertices[m.model_point_id - 1].x;
                feature_pt_2_data[3] = model.vertices[m.model_point_id - 1].y;
                feature_pt_2_data[4] = model.vertices[m.model_point_id - 1].z;
            }
            //       cout<<"m.feature_name  "<< m.feature_name <<endl;
            //      cout<<"m.feature_id = "<<m.feature_id<<endl;

        }//auto m
        if(dbg_flag == 1){
            printf("feature_pt_1_data[0]=%f, feature_pt_1_data[1]=%f,feature_pt_1_data[2]=%f,feature_pt_1_data[3]=%f,feature_pt_1_data[4]=%f\n",feature_pt_1_data[0],feature_pt_1_data[1],feature_pt_1_data[2],feature_pt_1_data[3],feature_pt_1_data[4]);
            printf("feature_pt_2_data[0]=%f, feature_pt_2_data[1]=%f,feature_pt_2_data[2]=%f,feature_pt_2_data[3]=%f,feature_pt_2_data[4]=%f\n",feature_pt_2_data[0],feature_pt_2_data[1],feature_pt_2_data[2],feature_pt_2_data[3],feature_pt_2_data[4]);
        }

        if(i <= 8){
            re_1[i][0] = 0;
            re_1[i][1] =  diff_calc(feature_pt_1_data, feature_pt_2_data);
        } else if(i <= 12){
            re_1[i][0] = 1;
            re_1[i][1] =  diff_calc(feature_pt_1_data, feature_pt_2_data);
        } else if(i <= 16){
            re_1[i][0] = 2;
            re_1[i][1] =  diff_calc(feature_pt_1_data, feature_pt_2_data);
        } else{
            re_1[i][0] = 3;
            re_1[i][1] =  diff_calc(feature_pt_1_data, feature_pt_2_data);
        }

        if(dbg_flag == 1)    cout<<"i = "<<i<<" re_1 = "<< re_1[i][0]<<" value "<<re_1[i][1] <<endl;
        //   if(outfile.is_open())
        //      outfile<<re_1[i][0]<<" "<<re_1[i][1]<< " "<<  feature_pt_1_data[0] <<" " << feature_pt_1_data[1] <<endl;
        arr[i][0] = re_1[i][0]; arr[i][1] = re_1[i][1];
    }// i
    //outfile.close();
}
//----------------------------------------------



void convert_to_grayscale(const cv::Mat& in, cv::Mat& out)
{
    if(in.channels() == 3)
    {
        // Make sure it's in a correct format
        if(in.depth() != CV_8U)
        {
            if(in.depth() == CV_16U)
            {
                cv::Mat tmp = in / 256;
                tmp.convertTo(tmp, CV_8U);
                cv::cvtColor(tmp, out, CV_BGR2GRAY);
            }
        }
        else
        {
            cv::cvtColor(in, out, CV_BGR2GRAY);
        }
    }
    else if(in.channels() == 4)
    {
        cv::cvtColor(in, out, CV_BGRA2GRAY);
    }
    else
    {
        if(in.depth() == CV_16U)
        {
            cv::Mat tmp = in / 256;
            out = tmp.clone();
        }
        else if(in.depth() != CV_8U)
        {
            in.convertTo(out, CV_8U);
        }
        else
        {
            out = in.clone();
        }
    }
}

//**************Scale Rotation Transform*****************************
Mesh::ObjIO scRotaTrans(int flag,int  reg_pt_flag, float reg_point_t0, float reg_point_t1, int model_point_id_nt, cv::Mat image, float trans_x0, float trans_x1, float trans_y0, float trans_y1, float scale,  float *trans_m,float * theta_xyz,  Mesh::ObjIO model){

    float  tx0, ty0, tx1, ty1, tx2, ty2;
    float vertex_m[4];

    for (int vertex_idx = 0; vertex_idx < model.vertices.size(); vertex_idx++) {
        model.vertices[vertex_idx] = model.vertices[vertex_idx] * scale;
        model.vertices[vertex_idx].w = 1;

        vertex_m[0] = model.vertices[vertex_idx].x;
        vertex_m[1] = model.vertices[vertex_idx].y;
        vertex_m[2] = model.vertices[vertex_idx].z;
        vertex_m[3] = model.vertices[vertex_idx].w;
        //2 rotation
        rotaion(theta_xyz, vertex_m);

        if((flag == 0)&&(reg_pt_flag > 4))
        {
            if(vertex_idx == reg_point_t0){
                tx0 = trans_x0 - vertex_m[0];
                ty0 = image.rows - trans_y0 - vertex_m[1];
            }
            if(vertex_idx == reg_point_t1){
                tx1 = trans_x1 - vertex_m[0];
                ty1 = image.rows - trans_y1 - vertex_m[1];
            }
        }else if((flag == 0)&&(reg_pt_flag < 4)){
            if (vertex_idx == model_point_id_nt) {
                trans_m[0] = trans_m[0] - vertex_m[0];
                trans_m[1] = image.rows - trans_m[1] - vertex_m[1];
            }
        }else{
            if (vertex_idx == model_point_id_nt) {
                trans_m[0] = trans_m[0] - vertex_m[0];
                trans_m[1] = image.rows - trans_m[1] - vertex_m[1];
            }
        }

        glm::tvec4<float> vec4_t(vertex_m[0], vertex_m[1], vertex_m[2], vertex_m[3]);;
        model.vertices[vertex_idx] = vec4_t;
    }

    if((flag == 0)&&(reg_pt_flag > 4)){
        trans_m[0] = (tx0 + tx1) / 2;
        trans_m[1] = (ty0 + ty1) / 2;
    }else{

    }

    printf("trans_m[0]=%f trans_m[1]=%f\n",trans_m[0],trans_m[1]);
    //3. translation
    for (int vertex_idx = 0; vertex_idx < model.vertices.size(); vertex_idx++) {
        model.vertices[vertex_idx].x = model.vertices[vertex_idx].x + trans_m[0];
        model.vertices[vertex_idx].y = model.vertices[vertex_idx].y + trans_m[1];
    }
    return model;

}


//***Openface fun generate headPose************
cv::Vec6d openFaceFun(cv::Mat image){

    LandmarkDetector::FaceModelParameters det_parameters;
    cv::Mat_<uchar> grayscale_image;
    convert_to_grayscale(image, grayscale_image);
    cout<<"grayscale_image = "<<grayscale_image.size()<<endl;

    det_parameters.model_location = "/home/tbb/Documents/DOC_GUO/Program/01_yrcpp/lib/local/LandmarkDetector/model/main_clnf_general.txt";
    vector<cv::Rect_<double> > face_detections;
    cout<<"face_detections.size()"<<face_detections.size()<<endl;
    dlib::frontal_face_detector face_detector_hog = dlib::get_frontal_face_detector();
    vector<double> confidences;
    LandmarkDetector::DetectFacesHOG(face_detections, grayscale_image, face_detector_hog, confidences);

    //printf("face_detections[0].size() = %d, face_detections[0].height = %d, face_detections[0].width = %d\n", face_detections[0].size(), face_detections[0].height, face_detections[0].width);

    cout<<"face_detections.size()"<<face_detections.size()<<endl;
    cv::Mat_<float> depth_image;
    LandmarkDetector::CLNF clnf_model(det_parameters.model_location);
    cout<<"clnf_model.size"<<sizeof(clnf_model)<<endl;
    cout<<"det_parameters.model_location"<< det_parameters.model_location <<endl;

    float fx = 0, fy = 0, cx = 0, cy = 0;
    fx = 500 * (grayscale_image.cols / 640.0);
    fy = 500 * (grayscale_image.rows / 480.0);

    fx = (fx + fy) / 2.0;
    fy = fx;

    cx = grayscale_image.cols / 2.0f;
    cy = grayscale_image.rows / 2.0f;

    // if there are multiple detections go through them
    bool success = LandmarkDetector::DetectLandmarksInImage(grayscale_image, depth_image, face_detections[0], clnf_model, det_parameters);
   // printf("fx = %f, fy =%f, cx=%f, cy=%f\n", fx, fy, cx, cy);
    // Estimate head pose and eye gaze
    cv::Vec6d headPose = LandmarkDetector::GetCorrectedPoseWorld(clnf_model, fx, fy, cx, cy);
    cout<<"headPose = "<<headPose<<endl;
    cout<<"det_parameters = "<<det_parameters.limit_pose<<endl;
    cout<<"**********************************Openface complete***********************"<<endl;
    return  headPose;
}
//***********get FacialFeatureUV************************************************
Mesh::ObjIO getFacialFeatureUV(int reg_pt_flag,cv::Mat image, Mesh::ObjIO model, float left_x, float right_x, float top_y, float bottom_y){
    Mesh::ObjIO model_new(true);
    float x_t, y_t;
    int count = 1;
    for (int i = 0; i < model.tvi.size(); i++) {
        const auto &triangle_indices = model.tvi[i];
        //printf("x_t = %f, y_t = %f\n", x_t, y_t);
        bool is_find = false;
        for (int k = 0; k < 3; k++) {
            x_t = model.vertices[triangle_indices[k] - 1].x;
            y_t = model.vertices[triangle_indices[k] - 1].y;

            if(reg_pt_flag < 4){
                if ((x_t > left_x) && (x_t < right_x) && (y_t > image.rows - bottom_y) && (y_t < image.rows - top_y)) {
                    is_find = true;
                    for (int j = 0; j < 3; j++) {
                        model_new.vertices.push_back(model.vertices[model.tvi[i][j] - 1]);
                        model_new.texcoords.push_back(model.texcoords[model.tci[i][j] - 1]);
                    }
                    model_new.tvi.push_back({count, count + 1, count + 2});
                    model_new.tci.push_back({count, count + 1, count + 2});
                    count += 3;
                    break;
                }//x_t
            }else{
                is_find = true;
                for (int j = 0; j < 3; j++) {
                    model_new.vertices.push_back(model.vertices[model.tvi[i][j] - 1]);
                    model_new.texcoords.push_back(model.texcoords[model.tci[i][j] - 1]);
                }
                model_new.tvi.push_back({count, count + 1, count + 2});
                model_new.tci.push_back({count, count + 1, count + 2});
                count += 3;
                break;
            }
        }//k
        if (is_find == true) {
            continue;
        }
    }
    return model_new;
}

//************getFacialFeatureBoder********************************************
void getWomanFacialFeatureBoder(float *boder, string reg_point, Mark::MapperCollection vec_map){

    float left_x, right_x, top_y, bottom_y;
    float  nose_tip_tran_x, nose_tip_tran_y,model_point_id_nt;

    if (reg_point == "left_eye_center") {
        for (auto &&m : vec_map) {
            if (m.feature_name == "left_eyebrow_left_corner") {
                left_x = m.pixel_x - 50;
            }
            if (m.feature_name == "left_eye_right_corner") {
                right_x = m.pixel_x + 50;
            }
            if (m.feature_name == "left_eyebrow_upper_middle") {
                top_y = m.pixel_y - 50;
            }
            if (m.feature_name == "left_eye_bottom") {
                bottom_y = m.pixel_y+50;
            }

            if (m.feature_name == "left_eye_right_corner") {
                nose_tip_tran_x = m.pixel_x;
                nose_tip_tran_y = m.pixel_y;
                model_point_id_nt = m.model_point_id - 1;
            }
        }
    }

    if (reg_point == "right_eye_center") {
        for (auto &&m : vec_map) {
            if (m.feature_name == "right_eyebrow_left_corner") {
                left_x = m.pixel_x -60;
            }
            if (m.feature_name == "right_eye_right_corner") {
                right_x = m.pixel_x+100;
            }
            if (m.feature_name == "right_eyebrow_upper_middle") {
                top_y = m.pixel_y-50;
            }
            if (m.feature_name == "right_eye_bottom") {
                bottom_y = m.pixel_y+50;
            }
            if (m.feature_name == "right_eye_left_corner") {
                nose_tip_tran_x = m.pixel_x;
                nose_tip_tran_y = m.pixel_y;
                model_point_id_nt = m.model_point_id - 1;
            }
        }
    }

    if (reg_point == "nose_tip") {
        for (auto &&m : vec_map) {
            if (m.feature_name == "nose_left") {
                left_x = m.pixel_x -200 ;
            }
            if (m.feature_name == "nose_right") {
                right_x = m.pixel_x +200;
            }
            if (m.feature_name == "nose_contour_left2") {
                top_y = m.pixel_y - 40;
            }
            if (m.feature_name == "nose_contour_lower_middle") {
                bottom_y = m.pixel_y + 50;
            }

            if (m.feature_name == "nose_contour_lower_middle") {
                nose_tip_tran_x = m.pixel_x;
                nose_tip_tran_y = m.pixel_y;
                model_point_id_nt = m.model_point_id - 1;
            }
        }
    }

    if (reg_point == "mouth_lower_lip_top") {
        for (auto &&m : vec_map) {
            if (m.feature_name == "mouth_left_corner") {
                left_x = m.pixel_x-200 ;
            }
            if (m.feature_name == "mouth_right_corner") {
                right_x = m.pixel_x+300 ;
            }
            if (m.feature_name == "mouth_upper_lip_top") {
                top_y = m.pixel_y - 30;
            }
            if (m.feature_name == "mouth_lower_lip_bottom") {
                bottom_y = m.pixel_y + 150;
            }

            if (m.feature_name == "mouth_upper_lip_bottom") {
                nose_tip_tran_x = m.pixel_x;
                nose_tip_tran_y = m.pixel_y;
                model_point_id_nt = m.model_point_id - 1;
            }
        }
    }

    boder[0] = left_x;
    boder[1] = right_x ;
    boder[2] = top_y ;
    boder[3] = bottom_y ;
    boder[4] = nose_tip_tran_x ;
    boder[5] = nose_tip_tran_y;
    boder[6] = model_point_id_nt ;
//    printf("boder[0]=%f, boder[1]=%f,boder[2]=%f,boder[3]=%f,boder[4]=%f,boder[5]=%f,boder[6]=%f\n", boder[0], boder[1],boder[2],boder[3],boder[4],boder[5],boder[6]);
}

void getmanFacialFeatureBoder(float *boder, string reg_point, Mark::MapperCollection vec_map){

    float left_x, right_x, top_y, bottom_y;
    float  nose_tip_tran_x, nose_tip_tran_y,model_point_id_nt;

    if (reg_point == "left_eye_center") {
        for (auto &&m : vec_map) {
            if (m.feature_name == "left_eyebrow_left_corner") {
                left_x = m.pixel_x - 100;
            }
            if (m.feature_name == "left_eye_right_corner") {
                right_x = m.pixel_x + 200;
            }
            if (m.feature_name == "left_eyebrow_upper_middle") {
                top_y = m.pixel_y - 400;
            }
            if (m.feature_name == "left_eye_bottom") {
                bottom_y = m.pixel_y+150;
            }

            if (m.feature_name == "left_eye_right_corner") {
                nose_tip_tran_x = m.pixel_x;
                nose_tip_tran_y = m.pixel_y;
                model_point_id_nt = m.model_point_id - 1;
            }
        }
    }

    if (reg_point == "right_eye_center") {
        for (auto &&m : vec_map) {
            if (m.feature_name == "right_eyebrow_left_corner") {
                left_x = m.pixel_x -100;
            }
            if (m.feature_name == "right_eye_right_corner") {
                right_x = m.pixel_x+200;
            }
            if (m.feature_name == "right_eyebrow_upper_middle") {
                top_y = m.pixel_y-400;
            }
            if (m.feature_name == "right_eye_bottom") {
                bottom_y = m.pixel_y+150;
            }
            if (m.feature_name == "right_eye_left_corner") {
                nose_tip_tran_x = m.pixel_x;
                nose_tip_tran_y = m.pixel_y;
                model_point_id_nt = m.model_point_id - 1;
            }
        }
    }

    if (reg_point == "nose_tip") {
        for (auto &&m : vec_map) {
            if (m.feature_name == "nose_left") {
                left_x = m.pixel_x -500 ;
            }
            if (m.feature_name == "nose_right") {
                right_x = m.pixel_x +500;
            }
            if (m.feature_name == "nose_contour_left2") {
                top_y = m.pixel_y - 100;
            }
            if (m.feature_name == "nose_contour_lower_middle") {
                bottom_y = m.pixel_y + 100;
            }

            if (m.feature_name == "nose_contour_lower_middle") {
                nose_tip_tran_x = m.pixel_x;
                nose_tip_tran_y = m.pixel_y;
                model_point_id_nt = m.model_point_id - 1;
            }
        }
    }

    if (reg_point == "mouth_lower_lip_top") {
        for (auto &&m : vec_map) {
            if (m.feature_name == "mouth_left_corner") {
                left_x = m.pixel_x-200 ;
            }
            if (m.feature_name == "mouth_right_corner") {
                right_x = m.pixel_x+300 ;
            }
            if (m.feature_name == "mouth_upper_lip_top") {
                top_y = m.pixel_y - 50;
            }
            if (m.feature_name == "mouth_lower_lip_bottom") {
                bottom_y = m.pixel_y + 150;
            }

            if (m.feature_name == "mouth_upper_lip_bottom") {
                nose_tip_tran_x = m.pixel_x;
                nose_tip_tran_y = m.pixel_y;
                model_point_id_nt = m.model_point_id - 1;
            }
        }
    }

    boder[0] = left_x;
    boder[1] = right_x ;
    boder[2] = top_y ;
    boder[3] = bottom_y ;
    boder[4] = nose_tip_tran_x ;
    boder[5] = nose_tip_tran_y;
    boder[6] = model_point_id_nt ;
//    printf("boder[0]=%f, boder[1]=%f,boder[2]=%f,boder[3]=%f,boder[4]=%f,boder[5]=%f,boder[6]=%f\n", boder[0], boder[1],boder[2],boder[3],boder[4],boder[5],boder[6]);
}
//**********getScale****************************
float getScale(int flag, float  *trans_point, Mark::MapperCollection vec_map,Mesh::ObjIO model_clone){

    float nose_tip_tran_x = 0,nose_tip_tran_y = 0;
    int model_point_id_nt = 0;
    float scale = 0.00, scale_t = 0;
    float scale01, scale02, scale12,scale23, x0, y0, x1, y1, x2, y2, x3, y3;
    float thd_x0,thd_y0,thd_z0,thd_x1,thd_y1,thd_z1,thd_x2,thd_y2,thd_z2, thd_x3,thd_y3,thd_z3;
    float trans_x0, trans_y0, trans_x1, trans_y1, reg_point_t0, reg_point_t1;
    string st0,st1, st2;

    float x_t0, y_t0, x_t1, y_t1, z_t0, z_t1;
    float thd_xt0,thd_xt1, thd_yt0, thd_yt1, thd_zt0, thd_zt1;
    string ts0 = "contour_left1", ts1 = "contour_right1";


    //tranlation offset according nose_tip
    if(flag == 0 || flag == 3){
        st0 = "left_eye_left_corner";//left_eye_left_corner left_eyebrow_upper_middle
        st1 = "right_eye_right_corner";//right_eye_right_corner  right_eyebrow_upper_middle
        st2 = "nose_tip";// mouth_lower_lip_top it seems the result is ok?
    }else if(flag == 1){
        st0 = "mouth_upper_lip_top";
        st1 = "contour_chin";
        st2 = "left_eye_right_corner";//left_eye_left_corner nose_tip
    }else{
        st0 = "mouth_upper_lip_top";
        st1 = "contour_chin";
        st2 = "right_eye_left_corner";//right_eye_left_corner
    }
    for (auto m : vec_map) {
        int vertex_idx = m.model_point_id - 1;

        if (m.feature_name == st0) { //mouth_upper_lip_top
            x0 = m.pixel_x;
            y0 = m.pixel_y;

            trans_x0 = m.pixel_x;
            trans_y0 = m.pixel_y;
            reg_point_t0 = m.model_point_id - 1;//registration according to this point

            thd_x0 = model_clone.vertices[m.model_point_id - 1].x;
            thd_y0 = model_clone.vertices[m.model_point_id - 1].y;
            thd_z0 = model_clone.vertices[m.model_point_id - 1].z;
        }
        if (m.feature_name == st1) {
            x1 = m.pixel_x;
            y1 = m.pixel_y;

            trans_x1 = m.pixel_x;
            trans_y1 = m.pixel_y;
            reg_point_t1 = m.model_point_id - 1;

            thd_x1 = model_clone.vertices[m.model_point_id - 1].x;
            thd_y1 = model_clone.vertices[m.model_point_id - 1].y;
            thd_z1 = model_clone.vertices[m.model_point_id - 1].z;
        }
        if (m.feature_name == st2) {
            x2 = m.pixel_x;
            y2 = m.pixel_y;

            model_point_id_nt = m.model_point_id - 1;//registration according to this point
            nose_tip_tran_x = m.pixel_x;
            nose_tip_tran_y = m.pixel_y;

            thd_x2 = model_clone.vertices[m.model_point_id - 1].x;
            thd_y2 = model_clone.vertices[m.model_point_id - 1].y;
            thd_z2 = model_clone.vertices[m.model_point_id - 1].z;
        }
        if (m.feature_name == "contour_chin") {//mouth_lower_lip_bottom
            x3 = m.pixel_x;
            y3 = m.pixel_y;
            thd_x3 = model_clone.vertices[m.model_point_id - 1].x;
            thd_y3 = model_clone.vertices[m.model_point_id - 1].y;
            thd_z3 = model_clone.vertices[m.model_point_id - 1].z;
        }

        if (m.feature_name == ts0){//mouth_lower_lip_bottom
            x_t0 = m.pixel_x;
            y_t0 = m.pixel_y;
            thd_xt0 = model_clone.vertices[m.model_point_id - 1].x;
            thd_yt0 = model_clone.vertices[m.model_point_id - 1].y;
            thd_zt0 = model_clone.vertices[m.model_point_id - 1].z;
        }

        if (m.feature_name == ts1){//mouth_lower_lip_bottom
            x_t1 = m.pixel_x;
            y_t1 = m.pixel_y;
            thd_xt1 = model_clone.vertices[m.model_point_id - 1].x;
            thd_yt1 = model_clone.vertices[m.model_point_id - 1].y;
            thd_zt1 = model_clone.vertices[m.model_point_id - 1].z;
        }

    }
//        printf("x0 = %f, y0 = %f, thd_x0 = %f, thd_y0 = %f, th_z0=%f\n", x0 ,y0,thd_x0,thd_y0,thd_z0);
//        printf("x1 = %f, y1 = %f, thd_x1 = %f, thd_y1 = %f, th_z1=%f\n", x1 ,y1,thd_x1,thd_y1,thd_z1);
//        printf("x2 = %f, y2 = %f, thd_x2 = %f, thd_y2 = %f, th_z2=%f\n", x2 ,y2,thd_x2,thd_y2,thd_z2);

    scale01 = sqrt((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1)) /  sqrt((thd_x0 - thd_x1) * (thd_x0 - thd_x1) + (thd_y0 - thd_y1) * (thd_y0 - thd_y1) );//+  (thd_z0 - thd_z1) * (thd_z0 - thd_z1)
    scale02 = sqrt((x0 - x2) * (x0 - x2) + (y0 - y2) * (y0 - y2)) /  sqrt((thd_x0 - thd_x2) * (thd_x0 - thd_x2) + (thd_y0 - thd_y2) * (thd_y0 - thd_y2) );//+  (thd_z0 - thd_z2) * (thd_z0 - thd_z2)
    scale12 = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)) /  sqrt((thd_x1 - thd_x2) * (thd_x1 - thd_x2) + (thd_y1 - thd_y2) * (thd_y1 - thd_y2) );//+  (thd_z1 - thd_z2) * (thd_z1 - thd_z2)
    scale = (scale01 + scale02 + scale12) / 3;

    scale_t = sqrt((x_t0 - x_t1) * (x_t0 - x_t1) + (y_t0 - y_t1) * (y_t0 - y_t1)) /  sqrt((thd_xt0 - thd_xt1) * (thd_xt0 - thd_xt1) + (thd_yt0 - thd_yt1) * (thd_yt0 - thd_yt1));
    if(flag == 0||flag == 3){
        scale = (scale01 + scale_t)/2;
    }
    printf("scale01=%f,  scale02=%f  scale12=%f scale23 = %f scale = %f  \n", scale01, scale02, scale12, scale23, scale);
    printf("scale = %f\n", scale);
    trans_point[0] = nose_tip_tran_x;
    trans_point[1] = nose_tip_tran_y;
    trans_point[2] = model_point_id_nt;
    trans_point[3] = reg_point_t0 ;
    trans_point[4] = reg_point_t1 ;
    trans_point[5] = trans_x0;
    trans_point[6] = trans_y0;
    trans_point[7] = trans_x1;
    trans_point[8] = trans_y1;

    printf("trans_point[0] = %f, trans_point[1] = %f,trans_point[2] = %f,trans_point[3] = %f,trans_point[4] = %f\n", trans_point[0], trans_point[1],trans_point[2] ,trans_point[3],trans_point[4]);
    return scale;
}

cv::Mat mixFacialFeatures(cv::Mat png_front_isomap,cv::Mat png_front_ind_isomap,string pathlefteyemask, string pathrighteyemask, string pathnosemask, string pathmouthmask){
  cv::Mat lefteyemask = cv::imread(pathlefteyemask);
  cv::Mat righteyemask= cv::imread(pathrighteyemask);
  cv::Mat nosemask= cv::imread(pathnosemask);
  cv::Mat mouthmask= cv::imread(pathmouthmask);

  cv::Mat output;
  output= render::possionImage(png_front_ind_isomap,  png_front_isomap, lefteyemask );
  output= render::possionImage(png_front_ind_isomap,  output,  righteyemask);
  output= render::possionImage(png_front_ind_isomap,  output,  nosemask);
  output= render::possionImage(png_front_ind_isomap,  output,  mouthmask);
  return output;
}

/***
 * Calculate the Uv Image
 * @param path_image the path of input per-calculate image
 * @param path_obj the path of obj file
 * @param path_uv_mapper the path of uv_mapper(contains the correspondence of 2D-landMark from Img and the 3D-landMark from Obj)
 * @param path_out the path of output file
 * @param is_draw_landmark if or not draw 2d-landMark
 * @param is_draw_wireframe if or not draw 3d projection to 2D-image  fs::path path_image
 * @return the projection image and the isomap image
 */
std::pair<cv::Mat,cv::Mat> generateUvImage(int gender, int flag, int lite_flag,  float arr[][2], float *theta, cv::Mat final_img,cv::Mat image, Mesh::ObjIO model, Mark::MapperCollection vec_map,
                                           bool is_draw_landmark = true, bool is_draw_wireframe = true) {

    cv::Mat out_img = image.clone();
    Mesh::ObjIO model_clone = model;
    cv::Mat isomap;
    std::cout << "Start generateUvImage " << flag << std::endl;

    float nose_tip_tran_x = 0,nose_tip_tran_y = 0;
    int model_point_id_nt = 0;
    float scale = 0.00;
    float scale01, scale02, scale12, x0, y0, x1, y1, x2, y2;
    float thd_x0,thd_y0,thd_z0,thd_x1,thd_y1,thd_z1,thd_x2,thd_y2,thd_z2;

    int dbg_flag = 0;
    int reg_pt_flag = 5; // 0: left_eye_center 1:right_eye_center 2: mouth other:nose_tip  or 0: nose tip, 1:left_eye_center 2:right_eye_center 3:mouth lower_lip_top
    float trans_x0, trans_y0, trans_x1, trans_y1, reg_point_t0, reg_point_t1;
    int is_draw_wire = 1;

    string reg_point = "";
    string st0,st1, st2;
    //************************************************Openface test************************************************************
      int openface =0 ;
      cv::Vec6d headPose;
      if(openface==1 ){
        headPose = openFaceFun(image);
      }
    //************************************************Openface test End************************************************************
    int reg_pt_flag_bg = 5, reg_pt_flag_ed = 6;
    if(flag == 3){
      reg_pt_flag_bg = 0;//from 0 to 4,left_eye, right_eye, nose, mouth 
      reg_pt_flag_ed = 4;
    }else{
    
      reg_pt_flag_bg = 5;//glob mapping
      reg_pt_flag_ed = 6;
    }
for(reg_pt_flag = reg_pt_flag_bg; reg_pt_flag < reg_pt_flag_ed; reg_pt_flag++){
    if (reg_pt_flag == 0) {
        reg_point = "nose_tip";
    } else if (reg_pt_flag == 1) {
        reg_point = "left_eye_center";
    } else if (reg_pt_flag == 2) {
        reg_point = "right_eye_center";
    } else if(reg_pt_flag == 3) {
        reg_point = "mouth_lower_lip_top";
    } else{
    
    }
//**************get scale************************************************************************************************
  float trans_point[9];
  scale =  getScale(flag, trans_point, vec_map, model_clone);
  nose_tip_tran_x = trans_point[0];
  nose_tip_tran_y = trans_point[1];
  model_point_id_nt =  trans_point[2];
  reg_point_t0 = trans_point[3];
  reg_point_t1 = trans_point[4];
  trans_x0 = trans_point[5];
  trans_y0 = trans_point[6];
  trans_x1 = trans_point[7];
  trans_y1 = trans_point[8];
  printf("trans_point[0] = %f, trans_point[1] = %f,trans_point[2] = %f,trans_point[3] = %f,trans_point[4] = %f\n", trans_point[0], trans_point[1],trans_point[2] ,trans_point[3],trans_point[4]);
//************get facial feature boder***********************************************************************************
  float boder[7];
  float left_x, right_x, top_y, bottom_y;

  if(reg_pt_flag>=0 && reg_pt_flag<=3){
      if(gender == 0){
         getWomanFacialFeatureBoder(boder, reg_point, vec_map);
      }else{
         getmanFacialFeatureBoder(boder, reg_point, vec_map);
      }
      left_x = boder[0];
      right_x= boder[1];
      top_y= boder[2];
      bottom_y= boder[3];
      nose_tip_tran_x = boder[4];
      nose_tip_tran_y  = boder[5];
      model_point_id_nt  = (int)boder[6];
  }
    float vertex_m[4];
    float trans_m[3];
    float theta_xyz[3];

    if(openface == 1){//use openface theta
        if(flag == 0){
        theta_xyz[0] = headPose[3];         theta_xyz[1] = -headPose[4];        theta_xyz[2] = -headPose[5];
       }else if(flag == 1){
        theta_xyz[0] = headPose[3];         theta_xyz[1] = -headPose[4];        theta_xyz[2] = -headPose[5];
       }else{
        theta_xyz[0] = headPose[3];         theta_xyz[1] = -headPose[4];        theta_xyz[2] = -headPose[5];
       }
    }else{//use facepp theta
        theta_xyz[0] = theta[0];        theta_xyz[1] = theta[1];       theta_xyz[2] = theta[2];
    } 


   trans_m[0] = nose_tip_tran_x;   trans_m[1] = nose_tip_tran_y;  trans_m[2] = 0;
   model = model_clone;
   model= scRotaTrans(flag, reg_pt_flag, reg_point_t0, reg_point_t1, model_point_id_nt, image, trans_x0, trans_x1, trans_y0, trans_y1, scale, trans_m, theta_xyz, model);

   cv::Mat landmark_img;
   landmark_img = out_img.clone();

   Mesh::ObjIO model_new(true);
   model_new =  getFacialFeatureUV(reg_pt_flag,image,  model,  left_x,  right_x,  top_y,  bottom_y);

     if((flag == 0)&& (lite_flag == 1)){
        register_diff(model, vec_map, arr);
	LOG(INFO) << "------------------------------------register diff complete!------------------------------------";
    }
    //-----------------------------------------------------------------------------------------------------------------
    string temp_out_path("/home/tbb/Documents/DOC_GUO/Program/01_yrcpp/uvTexel/bin/t/");

    fs::path front_png_file;
    cv::Mat image_wireframe = image.clone();
    draw_wireframe(image_wireframe, model_new);
    if (flag == 0) {
        front_png_file = temp_out_path + "front.png";
    } else if (flag == 1) {
        front_png_file = temp_out_path + "right.png";
    } else {
        front_png_file = temp_out_path + "left.png";
    }  
    if (flag == 3) {
	front_png_file = temp_out_path + "ind.png";
    }

   if(is_draw_wire== 1)    cv::imwrite(front_png_file.string(), image_wireframe);
    std::cout << "Start extract_texture" << std::endl;
    isomap = render::extract_texture(final_img, model_new, image);
    std::cout << "Start extract_texture end" << std::endl;
    LOG(INFO) << "generate uvImage Info successfully !";
}//reg_pt_flag
printf("***************************************************************frame  %d complete*********************************************************************************************\n", flag);

return {out_img,isomap};
}
