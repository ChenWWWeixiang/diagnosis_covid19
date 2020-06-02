import os
import time
import shutil
import numpy as np
import SimpleITK as sitk
from glob import glob


PR = "crop"
REFER = "lungsegs"
MASK = "lungsegs"
TYPE_U = ["lesions", "lungsegs"]
KEYS = ["images", "lesions", "lungsegs"]


def command_iteration(method):
    print("{0:3} = {1:10.5f}".format(method.GetOptimizerIteration(), method.GetMetricValue()))


if __name__ == "__main__":
    assert f"{REFER}" in KEYS and f"{MASK}" in KEYS
    prefix = "/mnt/data7/NCP_mp_CTs"
    flag=0
    # regristration parameters settings
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsJointHistogramMutualInformation()
    R.SetMetricSamplingPercentage(0.01)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([4, 2, 1])
    R.SetOptimizerAsGradientDescent(
        learningRate=0.1, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10
    )
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    R.SetInterpolator(sitk.sitkLinear)
    # R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    name_lst = glob(f"{prefix}/{PR}/{REFER}/*")
    name_lst.sort()

    if os.path.exists(f"{prefix}/registration_failure.txt"):
        with open(f"{prefix}/registration_failure.txt", "r") as fp:
            reg_failed_lst = [x.strip("\n") for x in fp.readlines()]
    else:
        reg_failed_lst = []

    for name_pre in name_lst:
        #txs_pre = name_pre.replace(f"{PR}/{REFER}", "txs")
        #if os.path.isdir(txs_pre) and len(glob(f"{txs_pre}/*.txt")) >= 2:
        #    continue
        phase_lst = glob(f"{name_pre}/*.mha")
        phase_lst.sort()

        # copy the first image, as fixed image
        for key in KEYS:
            tar_name = phase_lst[0].replace(PR, "reg_to_one").replace(REFER, key)
            os.makedirs(os.path.dirname(tar_name),exist_ok=True)
            #shutil.copyfile(phase_lst[0].replace(REFER, key), tar_name)

        reg_succed = np.ones(len(phase_lst)).astype(np.bool)
        # registration for the second and other images
        if flag==0:##reg all to this
            flag=1
            fix_img = sitk.ReadImage(phase_lst[0], sitk.sitkFloat32)
            fix_msk = sitk.ReadImage(phase_lst[0].replace(REFER, MASK), sitk.sitkFloat32)
            fix_origin = fix_img.GetOrigin()
        for move_idx, move_name in enumerate(phase_lst):
            print(f"=> processing {move_name}")

            tx_name = move_name.replace(f"{PR}/{REFER}", "txs_to_one").replace(".mha", ".txt")
            if os.path.exists(tx_name):
                # already have map parameters
                outTx = sitk.ReadTransform(tx_name)
            else:
                print("\t execute registration...", end="", flush=True)
                end = time.time()
                # not registrated, generate it
                os.makedirs(os.path.dirname(tx_name),exist_ok=True)
                # get regristration transformation
                move_img = sitk.ReadImage(move_name, sitk.sitkFloat32)
                move_msk = sitk.ReadImage(move_name.replace(REFER, MASK), sitk.sitkFloat32)
                move_img.SetOrigin(fix_origin)
                move_msk.SetOrigin(fix_origin)
                R.SetInitialTransform(
                    sitk.CenteredTransformInitializer(
                        fix_img,
                        move_img,
                        sitk.AffineTransform(3),
                        sitk.CenteredTransformInitializerFilter.MOMENTS,
                    )
                )
                outTx = R.Execute(fix_msk, move_msk)
                sitk.WriteTransform(outTx, tx_name)
                print(f"...{time.time()-end:.2f}s")
            # execute transformation
            print(f"\t transform all...", end="", flush=True)
            end = time.time()
            for key in KEYS:
                interpolator = sitk.sitkNearestNeighbor if key in TYPE_U else sitk.sitkLinear
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(fix_img)
                resampler.SetInterpolator(interpolator)
                resampler.SetDefaultPixelValue(0)
                resampler.SetTransform(outTx)

                tar_name = move_name.replace(PR, "reg_to_one").replace(REFER, key)
                if not os.path.exists(tar_name):
                    dtype = sitk.sitkUInt8 if key in TYPE_U else sitk.sitkInt16
                    move_img = sitk.ReadImage(move_name.replace(REFER, key))
                    move_img.SetOrigin(fix_origin)
                    moved_img = resampler.Execute(move_img)
                    moved_img = sitk.Cast(moved_img, dtype)
                    sitk.WriteImage(moved_img, tar_name, True)
                elif "lungsegs" in key:
                    moved_img = sitk.ReadImage(tar_name)

                if "lungsegs" in key:
                    dice_filter = sitk.LabelOverlapMeasuresImageFilter()
                    dice_filter.Execute(fix_img > 0.5, moved_img > 0.5)
                    if dice_filter.GetDiceCoefficient() < 0.85:
                        reg_succed[move_idx] = False
                        print(
                            f"Registration failed! The DSC is {dice_filter.GetDiceCoefficient():.2f}...",
                            end="",
                            flush=True,
                        )

            print(f"...{time.time()-end:.2f}s")

        if not np.all(reg_succed):
            reg_failed_lst.append(name_pre)

    with open(f"{prefix}/registration_failure.txt", "w") as fp:
        fp.writelines("\n".join(reg_failed_lst))
